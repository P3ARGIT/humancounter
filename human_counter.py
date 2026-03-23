import argparse
import collections
import datetime as dt
import json
import os
import sqlite3
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import cv2
from ultralytics import YOLO


TrackHistory = Dict[int, Deque[Tuple[int, int]]]
TrackSides = Dict[int, int]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Count people entering/leaving a room with YOLO + tracking."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source. Use 0 for webcam, or a video file path/stream URL.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Ultralytics detection model path/name.",
    )
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument(
        "--line-orientation",
        choices=["vertical", "horizontal"],
        default="vertical",
        help="Counting line orientation.",
    )
    parser.add_argument(
        "--line-position",
        type=float,
        default=0.5,
        help="Line position as frame ratio (0.0-1.0).",
    )
    parser.add_argument(
        "--line-points",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=None,
        help=(
            "Custom counting line using normalized coords in [0,1]. "
            "Example: --line-points 0.02 0.42 0.98 0.66"
        ),
    )
    parser.add_argument(
        "--enter-side",
        choices=["negative", "positive"],
        default="negative",
        help=(
            "Entry side for custom line mode. Use with --line-points. "
            "Side is based on line direction X1,Y1 -> X2,Y2."
        ),
    )
    parser.add_argument(
        "--enter-from",
        choices=["left", "right", "top", "bottom"],
        default="left",
        help="Which side is considered 'outside -> inside' for entry.",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        help="Ultralytics tracker config (e.g. bytetrack.yaml, botsort.yaml).",
    )
    parser.add_argument(
        "--save-output",
        type=str,
        default="",
        help="Optional output video path (e.g. runs/counting.mp4).",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/human_counter.db",
        help="SQLite database path for event logging.",
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        default="",
        help="Optional camera/location label stored in database.",
    )
    parser.add_argument(
        "--preview-frame-path",
        type=str,
        default="data/live_preview.jpg",
        help="Path to continuously write latest annotated frame for dashboard preview.",
    )
    parser.add_argument(
        "--preview-every-n-frames",
        type=int,
        default=5,
        help="Write preview image every N processed frames.",
    )
    parser.add_argument(
        "--show-trails",
        action="store_true",
        help="Draw short movement trails for tracked people.",
    )
    return parser


def parse_source(source: str):
    if source.isdigit():
        return int(source)
    return source


def get_line(frame_w: int, frame_h: int, orientation: str, ratio: float):
    ratio = max(0.05, min(0.95, ratio))
    if orientation == "vertical":
        x = int(frame_w * ratio)
        return (x, 0), (x, frame_h)
    y = int(frame_h * ratio)
    return (0, y), (frame_w, y)


def get_custom_line(frame_w: int, frame_h: int, line_points: Tuple[float, float, float, float]):
    x1, y1, x2, y2 = line_points
    p1 = (int(x1 * frame_w), int(y1 * frame_h))
    p2 = (int(x2 * frame_w), int(y2 * frame_h))
    return p1, p2


def compute_side(center: Tuple[int, int], orientation: str, line_ratio: float, frame_shape) -> int:
    h, w = frame_shape[:2]
    cx, cy = center
    if orientation == "vertical":
        line_x = int(w * line_ratio)
        return -1 if cx < line_x else 1
    line_y = int(h * line_ratio)
    return -1 if cy < line_y else 1


def compute_side_custom(center: Tuple[int, int], line_pt1: Tuple[int, int], line_pt2: Tuple[int, int]) -> int:
    cx, cy = center
    x1, y1 = line_pt1
    x2, y2 = line_pt2
    signed = (cx - x1) * (y2 - y1) - (cy - y1) * (x2 - x1)
    if signed < 0:
        return -1
    if signed > 0:
        return 1
    return 0


def crossing_direction(
    prev_side: int,
    current_side: int,
    orientation: str,
    enter_from: str,
    custom_mode: bool,
    enter_side: str,
) -> Optional[str]:
    if prev_side == current_side or prev_side == 0 or current_side == 0:
        return None

    if custom_mode:
        if prev_side == -1 and current_side == 1:
            return "IN" if enter_side == "negative" else "OUT"
        if prev_side == 1 and current_side == -1:
            return "IN" if enter_side == "positive" else "OUT"
        return None

    if orientation == "vertical":
        enter_expected = "left"
        if enter_from == "right":
            enter_expected = "right"

        if prev_side == -1 and current_side == 1:
            return "IN" if enter_expected == "left" else "OUT"
        if prev_side == 1 and current_side == -1:
            return "IN" if enter_expected == "right" else "OUT"

    if orientation == "horizontal":
        enter_expected = "top"
        if enter_from == "bottom":
            enter_expected = "bottom"

        if prev_side == -1 and current_side == 1:
            return "IN" if enter_expected == "top" else "OUT"
        if prev_side == 1 and current_side == -1:
            return "IN" if enter_expected == "bottom" else "OUT"

    return None


def validate_enter_from(orientation: str, enter_from: str):
    if orientation == "vertical" and enter_from in {"top", "bottom"}:
        raise ValueError("For vertical line, --enter-from must be left or right")
    if orientation == "horizontal" and enter_from in {"left", "right"}:
        raise ValueError("For horizontal line, --enter-from must be top or bottom")


def validate_line_points(line_points: Optional[Tuple[float, float, float, float]]):
    if line_points is None:
        return
    x1, y1, x2, y2 = line_points
    values = (x1, y1, x2, y2)
    if not all(0.0 <= v <= 1.0 for v in values):
        raise ValueError("--line-points values must be normalized in [0.0, 1.0]")
    if x1 == x2 and y1 == y2:
        raise ValueError("--line-points endpoints must be different")


def init_db(db_path: str) -> sqlite3.Connection:
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_file))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            source TEXT NOT NULL,
            camera_name TEXT,
            model TEXT NOT NULL,
            line_config_json TEXT NOT NULL,
            in_count INTEGER NOT NULL DEFAULT 0,
            out_count INTEGER NOT NULL DEFAULT 0,
            inside_count INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            event_time TEXT NOT NULL,
            track_id INTEGER NOT NULL,
            direction TEXT NOT NULL,
            center_x INTEGER NOT NULL,
            center_y INTEGER NOT NULL,
            in_count INTEGER NOT NULL,
            out_count INTEGER NOT NULL,
            inside_count INTEGER NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
        """
    )
    conn.commit()
    return conn


def create_session(
    conn: sqlite3.Connection,
    source: str,
    camera_name: str,
    model: str,
    line_config: dict,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO sessions (started_at, source, camera_name, model, line_config_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            dt.datetime.now().isoformat(),
            source,
            camera_name,
            model,
            json.dumps(line_config),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def update_session_counts(conn: sqlite3.Connection, session_id: int, in_count: int, out_count: int):
    conn.execute(
        """
        UPDATE sessions
        SET in_count = ?, out_count = ?, inside_count = ?
        WHERE id = ?
        """,
        (in_count, out_count, in_count - out_count, session_id),
    )
    conn.commit()


def close_session(conn: sqlite3.Connection, session_id: int):
    conn.execute(
        """
        UPDATE sessions
        SET ended_at = ?
        WHERE id = ?
        """,
        (dt.datetime.now().isoformat(), session_id),
    )
    conn.commit()


def log_event(
    conn: sqlite3.Connection,
    session_id: int,
    track_id: int,
    direction: str,
    center_x: int,
    center_y: int,
    in_count: int,
    out_count: int,
):
    conn.execute(
        """
        INSERT INTO events (
            session_id,
            event_time,
            track_id,
            direction,
            center_x,
            center_y,
            in_count,
            out_count,
            inside_count
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            dt.datetime.now().isoformat(),
            track_id,
            direction,
            center_x,
            center_y,
            in_count,
            out_count,
            in_count - out_count,
        ),
    )
    conn.commit()


def main():
    args = build_parser().parse_args()
    validate_line_points(args.line_points)

    custom_mode = args.line_points is not None
    if not custom_mode:
        validate_enter_from(args.line_orientation, args.enter_from)

    model = YOLO(args.model)
    source = parse_source(args.source)
    source_text = str(args.source)

    if custom_mode:
        line_config = {
            "mode": "custom",
            "line_points": args.line_points,
            "enter_side": args.enter_side,
        }
    else:
        line_config = {
            "mode": "axis_aligned",
            "line_orientation": args.line_orientation,
            "line_position": args.line_position,
            "enter_from": args.enter_from,
        }

    db_conn = init_db(args.db_path)
    session_id = create_session(
        db_conn,
        source=source_text,
        camera_name=args.camera_name,
        model=args.model,
        line_config=line_config,
    )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    writer = None
    if args.save_output:
        output_path = Path(args.save_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_w, frame_h))

    preview_frame_path = Path(args.preview_frame_path)
    preview_frame_path.parent.mkdir(parents=True, exist_ok=True)
    preview_every_n_frames = max(1, int(args.preview_every_n_frames))
    frame_index = 0
    preview_tmp_path = preview_frame_path.with_name(
        f"{preview_frame_path.stem}.tmp{preview_frame_path.suffix}"
    )
    preview_write_warned = False
    preview_lock_warned = False

    history: TrackHistory = {}
    track_sides: TrackSides = {}

    session_in_count = 0
    session_out_count = 0
    daily_in_count = 0
    daily_out_count = 0
    current_day = dt.datetime.now().date()

    print("Press 'q' to quit.")
    print(f"Database: {args.db_path} | session_id={session_id}")
    print(f"Preview frame file: {preview_frame_path} (every {preview_every_n_frames} frames)")
    if custom_mode:
        x1, y1, x2, y2 = args.line_points
        print(
            "Counting line: custom "
            f"({x1:.2f},{y1:.2f}) -> ({x2:.2f},{y2:.2f}), enter_side={args.enter_side}"
        )
    else:
        print(
            f"Counting line: {args.line_orientation} at {args.line_position:.2f}, enter_from={args.enter_from}"
        )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1

            now = dt.datetime.now()
            if now.date() != current_day:
                close_session(db_conn, session_id)
                session_id = create_session(
                    db_conn,
                    source=source_text,
                    camera_name=args.camera_name,
                    model=args.model,
                    line_config=line_config,
                )
                current_day = now.date()
                session_in_count = 0
                session_out_count = 0
                daily_in_count = 0
                daily_out_count = 0
                print(f"{now.isoformat()} | Daily counters reset and new session started: {session_id}")

            results = model.track(
                source=frame,
                persist=True,
                classes=[0],  # person class only
                conf=args.conf,
                iou=args.iou,
                tracker=args.tracker,
                verbose=False,
            )

            result = results[0] if results else None
            annotated = frame.copy()

            if custom_mode:
                line_pt1, line_pt2 = get_custom_line(
                    annotated.shape[1],
                    annotated.shape[0],
                    tuple(args.line_points),
                )
            else:
                line_pt1, line_pt2 = get_line(
                    annotated.shape[1],
                    annotated.shape[0],
                    args.line_orientation,
                    args.line_position,
                )
            cv2.line(annotated, line_pt1, line_pt2, (0, 255, 255), 2)

            boxes = result.boxes if result is not None else None
            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                track_ids = boxes.id.int().cpu().tolist()
                confs = boxes.conf.cpu().tolist() if boxes.conf is not None else [0.0] * len(track_ids)

                for box, track_id, conf_score in zip(xyxy, track_ids, confs):
                    x1, y1, x2, y2 = map(int, box)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    if custom_mode:
                        current_side = compute_side_custom((cx, cy), line_pt1, line_pt2)
                    else:
                        current_side = compute_side(
                            (cx, cy), args.line_orientation, args.line_position, annotated.shape
                        )
                    previous_side = track_sides.get(track_id)
                    if previous_side is not None:
                        direction = crossing_direction(
                            previous_side,
                            current_side,
                            args.line_orientation,
                            args.enter_from,
                            custom_mode,
                            args.enter_side,
                        )
                        if direction == "IN":
                            session_in_count += 1
                            daily_in_count += 1
                            log_event(
                                db_conn,
                                session_id,
                                track_id,
                                "IN",
                                cx,
                                cy,
                                session_in_count,
                                session_out_count,
                            )
                            update_session_counts(db_conn, session_id, session_in_count, session_out_count)
                            print(f"{dt.datetime.now().isoformat()} | ID {track_id} | IN")
                        elif direction == "OUT":
                            session_out_count += 1
                            daily_out_count += 1
                            log_event(
                                db_conn,
                                session_id,
                                track_id,
                                "OUT",
                                cx,
                                cy,
                                session_in_count,
                                session_out_count,
                            )
                            update_session_counts(db_conn, session_id, session_in_count, session_out_count)
                            print(f"{dt.datetime.now().isoformat()} | ID {track_id} | OUT")

                    track_sides[track_id] = current_side

                    trail = history.setdefault(track_id, collections.deque(maxlen=24))
                    trail.append((cx, cy))

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (70, 220, 70), 2)
                    cv2.circle(annotated, (cx, cy), 4, (0, 120, 255), -1)
                    cv2.putText(
                        annotated,
                        f"ID {track_id} {conf_score:.2f}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                    )

                    if args.show_trails and len(trail) > 1:
                        pts = list(trail)
                        for idx in range(1, len(pts)):
                            cv2.line(annotated, pts[idx - 1], pts[idx], (255, 180, 0), 2)

            occupancy = max(0, session_in_count - session_out_count)
            cv2.putText(
                annotated,
                f"CURRENTLY INSIDE: {occupancy}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 200, 255),
                2,
            )
            cv2.putText(
                annotated,
                f"TODAY IN: {daily_in_count}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (30, 220, 30),
                2,
            )
            cv2.putText(
                annotated,
                f"TODAY OUT: {daily_out_count}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (20, 20, 230),
                2,
            )

            cv2.imshow("YOLO Human Counter", annotated)
            if writer is not None:
                writer.write(annotated)

            if frame_index % preview_every_n_frames == 0:
                ok_write = cv2.imwrite(str(preview_tmp_path), annotated)
                if ok_write:
                    try:
                        os.replace(preview_tmp_path, preview_frame_path)
                    except PermissionError:
                        # On Windows, dashboard readers can momentarily lock the destination file.
                        alt_path = preview_frame_path.with_name(
                            f"{preview_frame_path.stem}_{frame_index % 2}{preview_frame_path.suffix}"
                        )
                        cv2.imwrite(str(alt_path), annotated)
                        if not preview_lock_warned:
                            print(
                                "Warning: preview file lock detected; writing fallback files "
                                f"like {alt_path.name}"
                            )
                            preview_lock_warned = True
                    except OSError:
                        if not preview_write_warned:
                            print(f"Warning: could not update preview frame at {preview_frame_path}")
                            preview_write_warned = True
                elif not preview_write_warned:
                    print(f"Warning: could not write preview frame to {preview_frame_path}")
                    preview_write_warned = True

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        close_session(db_conn, session_id)
        db_conn.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
