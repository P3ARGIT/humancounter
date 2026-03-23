import argparse
import sqlite3


def main():
    parser = argparse.ArgumentParser(description="Inspect human counter SQLite data")
    parser.add_argument("--db-path", default="data/human_counter.db", help="Path to SQLite DB")
    parser.add_argument("--sessions-limit", type=int, default=5)
    parser.add_argument("--events-limit", type=int, default=10)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db_path)
    cur = conn.cursor()

    tables = cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    print("tables:", tables)

    sessions = cur.execute(
        """
        SELECT id, started_at, ended_at, source, camera_name, in_count, out_count, inside_count
        FROM sessions
        ORDER BY id DESC
        LIMIT ?
        """,
        (args.sessions_limit,),
    ).fetchall()
    print("\nrecent sessions:")
    for row in sessions:
        print(row)

    events = cur.execute(
        """
        SELECT id, session_id, event_time, track_id, direction, center_x, center_y, inside_count
        FROM events
        ORDER BY id DESC
        LIMIT ?
        """,
        (args.events_limit,),
    ).fetchall()
    print("\nrecent events:")
    for row in events:
        print(row)

    conn.close()


if __name__ == "__main__":
    main()
