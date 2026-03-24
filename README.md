# Human Counter (YOLO + Tracking)

This project counts people **entering** and **leaving** a room using a pretrained Ultralytics YOLO model and multi-object tracking.

## Features

- Detects only `person` class (COCO class `0`)
- Uses `YOLO.track(..., persist=True)` for stable person IDs across frames
- Counts crossings of a configurable virtual line
- Reports `IN`, `OUT`, and estimated `INSIDE = IN - OUT`
- Supports webcam or video file input

## 1) Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2) Run

### Webcam

```powershell
python human_counter.py --source 1 --line-points 0.00 0.48 1.00 0.66 --enter-side negative --camera-name FrontDoor --preview-frame-path live_preview.jpg --preview-every-n-frames 1
```

### Video file

```powershell
python human_counter.py --source .\videos\door.mp4 --model yolov8n.pt --line-orientation horizontal --line-position 0.55 --enter-from top
```

## Important parameters

- `--model`: Pretrained model name/path (example: `yolov8n.pt`, `yolo11n.pt`)
- `--source`: `0` for webcam, or a file/stream path
- `--line-orientation`: `vertical` or `horizontal`
- `--line-position`: Line location ratio in frame (`0.0` to `1.0`, practical range `0.2-0.8`)
- `--line-points X1 Y1 X2 Y2`: Custom line with normalized points in `[0,1]` (overrides orientation/position)
- `--enter-side`: With custom line mode, choose `negative` or `positive` as entry direction
- `--enter-from`: Defines which side means entry
  - Vertical line: `left` or `right`
  - Horizontal line: `top` or `bottom`
- `--show-trails`: Draw movement trails
- `--save-output`: Save annotated video (example: `runs\counted.mp4`)
- `--db-path`: SQLite database file path (default: `data/human_counter.db`)
- `--camera-name`: Optional label for camera/location saved with each session

## Example with saved output

```powershell
python human_counter.py --source .\videos\door.mp4 --show-trails --save-output runs\counted.mp4
```

## Example for a diagonal doorway line

Use this when your doorway boundary is slanted (like your red reference line):

```powershell
python human_counter.py --source 1 --line-points 0.02 0.42 0.98 0.66 --enter-side negative
```

If IN/OUT is reversed, keep the same line points and switch `--enter-side` to `positive`.

## Database logging

The app automatically saves data to SQLite each run:

- `sessions` table: one row per run/session
- `events` table: one row per `IN`/`OUT` crossing event

Example run with custom SQLite DB path and camera label:

```powershell
python human_counter.py --source 1 --camera-name FrontDoor --db-path data/human_counter.db
```

### Use shared Postgres for live cloud dashboard

Set `DATABASE_URL` in both places:

1. Render Web Service environment variable (`DATABASE_URL`) pointing to your Render Postgres.
2. Local counter machine environment variable (`DATABASE_URL`) with the same value.

When `DATABASE_URL` is set, both `human_counter.py` and `dashboard.py` use Postgres automatically.
`--db-path` remains as a SQLite fallback only when `DATABASE_URL` is not set.

PowerShell example (local counter machine):

```powershell
$env:DATABASE_URL="postgresql://USER:PASSWORD@HOST:PORT/DBNAME?sslmode=require"
python human_counter.py --source 1 --camera-name FrontDoor --line-points 0.00 0.48 1.00 0.66 --enter-side negative
```

Example run with diagonal line + dashboard live preview image export:

```powershell
python human_counter.py --source 1 --line-points 0.00 0.48 1.00 0.66 --enter-side negative --camera-name FrontDoor --db-path data/human_counter.db --preview-frame-path data/live_preview.jpg --preview-every-n-frames 5
```

Inspect latest events:

```powershell
python -c "import sqlite3; c=sqlite3.connect('data/human_counter.db'); print(c.execute('SELECT id, event_time, track_id, direction, in_count, out_count, inside_count FROM events ORDER BY id DESC LIMIT 10').fetchall())"
```

Inspect session summary:

```powershell
python -c "import sqlite3; c=sqlite3.connect('data/human_counter.db'); print(c.execute('SELECT id, started_at, ended_at, source, camera_name, in_count, out_count, inside_count FROM sessions ORDER BY id DESC LIMIT 5').fetchall())"
```

## Tuning tips

1. Place camera high enough to reduce occlusion.
2. Align the counting line with the actual doorway boundary.
3. Use a slightly larger model (`yolov8s.pt`) when detections are unstable.
4. Adjust confidence with `--conf` (default `0.35`) based on lighting and crowd density.

## Notes

- On first run, Ultralytics downloads pretrained weights automatically.
- Press `q` to quit.
- For 24/7 runs, daily counters automatically reset at `00:00` (midnight) without restarting the app.
- For 24/7 runs, the app also closes the current DB session and starts a new session automatically at `00:00`.

## Web Dashboard

A clean interactive dashboard is included to analyze saved SQLite data.

Install dashboard dependencies (already included in `requirements.txt`):

```powershell
pip install -r requirements.txt
```

Launch dashboard:

```powershell
python -m streamlit run dashboard.py
```

### Secure access (password protected)

The dashboard now supports built-in login protection.

1. Generate a password hash:

```powershell
python generate_password_hash.py MyStrongPassword123!
```

2. Set environment variables before launching dashboard:

```powershell
$env:DASHBOARD_AUTH_ENABLED="true"
$env:DASHBOARD_AUTH_USER="admin"
$env:DASHBOARD_PASSWORD_HASH="pbkdf2_sha256$260000$...$..."
python -m streamlit run dashboard.py
```

Notes:

- Authentication is enabled by default.
- If `DASHBOARD_PASSWORD_HASH` is missing while auth is enabled, dashboard access is blocked.
- Set `DASHBOARD_AUTH_ENABLED=false` only for trusted local development.

What you get:

- KPI cards (`Total IN`, `Total OUT`, `Net Inside`, `Sessions`)
- Daily and hourly charts
- Occupancy-over-time graph
- Direction split pie chart
- Session and event tables
- Sidebar filters (date range, camera, direction, session IDs)
- Nintendo-branded sidebar header (`Nintendo x Total Concept` logos)
- Nintendo Light visual style (single layout/theme)
- Optional live preview module (can be disabled to reduce load)
- Separate refresh controls for data and preview

Sidebar accordion order:

1. `Filters`
2. `Realtime`
3. `Data Source`

By default it reads:

- `data/human_counter.db` (SQLite fallback)

If `DATABASE_URL` is set, dashboard uses Postgres instead and ignores the SQLite path field.

You can change the DB path directly in the dashboard sidebar.

For live updates, keep both running:

1. `python human_counter.py ... --preview-frame-path data/live_preview.jpg`
2. `python -m streamlit run dashboard.py`

Then enable `Auto refresh data` and optionally `Enable preview module` in the dashboard sidebar.

## Deploy As Website

This project includes container files so you can deploy it as a web app.

### Local Docker test

```powershell
docker build -t human-counter-dashboard .
docker run --rm -p 8501:8501 human-counter-dashboard
```

Open: `http://localhost:8501`

### Cloud deployment options

- Azure Container Apps
- Azure App Service (custom container)
- AWS ECS/Fargate
- Render / Railway / any Docker-capable host

Deployment notes:

- For real-time cloud dashboard data, set the same `DATABASE_URL` on both local counter and cloud dashboard.
- If you stay on SQLite mode, mount/persist your `data/` folder so SQLite data survives restarts.
- Preview images are local files; cloud dashboard preview needs a shared image path or object storage sync.

DATABASE:$env:DATABASE_URL="postgresql://humancounter_db_user:uYCgjU675BnecE1Jh6ff2DKcV98LOwzi@dpg-d70ljhea2pns73b7sl8g-a.frankfurt-postgres.render.com/humancounter_db?sslmode=require"   
