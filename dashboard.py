from __future__ import annotations

import sqlite3
import base64
import hashlib
import hmac
import logging
import os
from io import BytesIO
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
try:
    import extra_streamlit_components as stx
except ImportError:  # pragma: no cover - optional at local dev time
    stx = None
from streamlit_autorefresh import st_autorefresh

try:
    import psycopg2
except ImportError:  # pragma: no cover - optional at local dev time
    psycopg2 = None


DEFAULT_DB_PATH = "data/human_counter.db"
DEFAULT_PREVIEW_PATH = "data/live_preview.jpg"
DEFAULT_NINTENDO_LOGO = "assets/nintendo-logo.png"
DEFAULT_TOTAL_CONCEPT_LOGO = "assets/total-concept-logo.png"
DEFAULT_AUTH_USER = "admin"
DEFAULT_AUTH_ITERATIONS = 260000
DEFAULT_AUTH_REMEMBER_DAYS = 30
AUTH_COOKIE_NAME = "human_counter_auth"
AUTH_QUERY_PARAM = "auth_token"
LIGHT_THEME = {
    "bg": "#fff6f6",
    "surface": "#ffffff",
    "text": "#1a1a1a",
    "muted": "#5e5e5e",
    "accent": "#e60012",
    "accent_alt": "#b9000f",
    "border": "#ffd7db",
    "template": "plotly_white",
    "in_color": "#00a651",
    "out_color": "#e60012",
}


def suppress_streamlit_media_logs() -> None:
    noisy_loggers = [
        "streamlit.runtime.memory_media_file_storage",
        "streamlit.web.server.media_file_handler",
    ]
    for logger_name in noisy_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False


def apply_theme_css(theme: dict[str, str]) -> None:
    st.markdown(
        f"""
<style>
:root {{
    --bg: {theme['bg']};
    --surface: {theme['surface']};
    --text: {theme['text']};
    --muted: {theme['muted']};
    --accent: {theme['accent']};
    --accent-alt: {theme['accent_alt']};
    --border: {theme['border']};
}}

[data-testid="stAppViewContainer"] {{
    background: radial-gradient(circle at 90% 0%, rgba(230,0,18,0.18), transparent 35%), var(--bg);
}}

[data-testid="stHeader"] {{
    display: none;
}}

[data-testid="stToolbar"] {{
    display: none;
}}

/* Keep desktop clean, but show the sidebar toggle/header on phones. */
@media (max-width: 900px) {{
    [data-testid="stHeader"] {{
        display: flex !important;
        background: #ffffff !important;
        border-bottom: 1px solid var(--border);
    }}

    [data-testid="stToolbar"] {{
        display: flex !important;
        background: transparent !important;
    }}

    .block-container {{
        padding-top: 0.5rem !important;
    }}
}}

[data-testid="stElementToolbar"] {{
    background: #ffffff !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}}

[data-testid="stElementToolbar"] button,
.stBaseButton-elementToolbar {{
    background: #ffffff !important;
    color: #1a1a1a !important;
    border-color: var(--border) !important;
}}

[data-testid="stElementToolbar"] button svg,
.stBaseButton-elementToolbar svg {{
    fill: #1a1a1a !important;
}}

.block-container {{
    padding-top: 1rem !important;
}}

[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, var(--surface), color-mix(in srgb, var(--surface), #000 12%));
    border-right: 1px solid var(--border);
}}

[data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"] {{
    padding-top: 0 !important;
}}

[data-testid="stSidebarHeader"] {{
    display: none !important;
}}

[data-testid="stSidebar"] .block-container {{
    /* Mirror body spacing and reserve bottom room for anchored logout button. */
    padding: 1rem 1rem 4.5rem 1rem !important;
}}

[data-testid="stSidebarUserContent"] {{
    position: relative;
    min-height: 100vh;
}}

h1, h2, h3, h4, h5, h6, p, span, label, div {{
    color: var(--text);
}}

.brand-x {{
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--accent);
    margin-top: 0.3rem;
    text-align: center;
}}

.dashboard-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.6rem;
}}

[data-testid="stExpander"] {{
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface);
}}

[data-testid="stExpander"] details {{
    background: var(--surface);
}}

[data-testid="stExpander"] summary {{
    background: var(--surface) !important;
    color: var(--text) !important;
    border-radius: 10px;
}}

[data-testid="stExpander"] summary p {{
    color: var(--text) !important;
    font-weight: 600;
}}

[data-testid="stExpander"] details[open] summary {{
    background: var(--surface) !important;
    color: var(--text) !important;
    border-bottom: 1px solid var(--border);
}}

div[data-testid="stMetric"] {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.5rem 0.7rem;
}}

button[kind="primary"] {{
    background: var(--accent) !important;
    border: 1px solid var(--accent) !important;
}}

[data-testid="stSidebar"] button[kind="secondary"] {{
    background: #1c2436 !important;
    border: 1px solid #2f3b54 !important;
    color: #ffffff !important;
}}

[data-testid="stSidebar"] button[kind="primary"] {{
    min-width: 96px;
}}

[data-testid="stSidebar"] button[kind="primary"]:not([disabled]) {{
    background: var(--accent) !important;
}}

[data-testid="stSidebar"] div[data-testid="stButton"]:has(button[kind="primary"]) {{
    position: absolute;
    left: 1rem;
    bottom: 1rem;
    z-index: 1100;
    margin: 0 !important;
    padding: 0 !important;
    background: transparent;
    width: fit-content !important;
    max-width: 160px;
}}

[data-testid="stSidebar"] div[data-testid="stButton"]:has(button[kind="primary"]) button {{
    width: auto !important;
}}

[data-testid="stSidebar"] button[kind="secondary"] * {{
    color: #ffffff !important;
}}

[data-testid="stSidebar"] [data-baseweb="select"] > div {{
    background: #070d1a !important;
    border-color: #ff8f99 !important;
    color: #ffffff !important;
}}

[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] div,
[data-testid="stSidebar"] [data-baseweb="select"] svg {{
    color: #ffffff !important;
    fill: #ffffff !important;
}}

[data-testid="stSidebar"] [role="listbox"] {{
    background: #070d1a !important;
    color: #ffffff !important;
}}

[data-testid="stSidebar"] [role="option"] {{
    background: #070d1a !important;
    color: #ffffff !important;
}}

[data-testid="stSidebar"] [role="option"][aria-selected="true"] {{
    background: #1a253d !important;
    color: #ffffff !important;
}}

[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {{
    color: #ffffff !important;
}}

/* Streamlit select dropdowns render in a portal outside sidebar; style globally. */
[data-baseweb="popover"] [role="listbox"] {{
    background: #070d1a !important;
    color: #ffffff !important;
}}

[data-baseweb="popover"] [role="option"] {{
    background: #070d1a !important;
    color: #ffffff !important;
}}

[data-baseweb="popover"] [role="option"] * {{
    color: #ffffff !important;
}}

[data-baseweb="popover"] [role="option"][aria-selected="true"],
[data-baseweb="popover"] [role="option"][aria-selected="true"] * {{
    background: #1a253d !important;
    color: #ffffff !important;
}}

[data-baseweb="popover"] [role="option"][aria-disabled="true"],
[data-baseweb="popover"] [role="option"][aria-disabled="true"] * {{
    color: #aeb6c9 !important;
}}
</style>
        """,
        unsafe_allow_html=True,
    )


def resolve_latest_preview(preview_path: Path) -> Path | None:
    candidates = []
    if preview_path.exists():
        candidates.append(preview_path)

    fallback_glob = f"{preview_path.stem}_*{preview_path.suffix}"
    candidates.extend([p for p in preview_path.parent.glob(fallback_glob) if p.is_file()])

    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


def read_image_bytes(image_path: Path) -> bytes | None:
    try:
        return image_path.read_bytes()
    except OSError:
        return None


def parse_env_bool(value: str, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_password_hash(stored: str):
    parts = stored.split("$")
    if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
        return None
    try:
        iterations = int(parts[1])
        salt = bytes.fromhex(parts[2])
        digest = bytes.fromhex(parts[3])
    except ValueError:
        return None
    return iterations, salt, digest


def verify_password(password: str, stored_hash: str) -> bool:
    parsed = parse_password_hash(stored_hash)
    if parsed is None:
        return False
    iterations, salt, expected_digest = parsed
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(candidate, expected_digest)


def is_auth_enabled() -> bool:
    return parse_env_bool(os.getenv("DASHBOARD_AUTH_ENABLED"), default=True)


def parse_remember_days() -> int:
    raw_value = os.getenv("DASHBOARD_AUTH_REMEMBER_DAYS", str(DEFAULT_AUTH_REMEMBER_DAYS))
    try:
        days = int(raw_value)
    except ValueError:
        return DEFAULT_AUTH_REMEMBER_DAYS
    return max(1, min(365, days))


def get_auth_secret(required_user: str, stored_hash: str) -> str:
    secret = os.getenv("DASHBOARD_AUTH_SECRET", "").strip()
    if secret:
        return secret
    # Backward-compatible fallback so existing deployments work without an extra env var.
    return hashlib.sha256(f"{required_user}|{stored_hash}".encode("utf-8")).hexdigest()


def make_auth_token(required_user: str, secret: str, remember_days: int) -> tuple[str, datetime]:
    now_utc = datetime.now(timezone.utc)
    expires_ts = int((now_utc + timedelta(days=remember_days)).timestamp())
    payload = f"{required_user}|{expires_ts}"
    signature = hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    token = base64.urlsafe_b64encode(f"{payload}|{signature}".encode("utf-8")).decode("ascii")
    # Cookie manager expects a local datetime for browser cookie expiration.
    cookie_expires_at = datetime.now() + timedelta(days=remember_days)
    return token, cookie_expires_at


def verify_auth_token(token: str, required_user: str, secret: str) -> bool:
    try:
        decoded = base64.urlsafe_b64decode(token.encode("ascii")).decode("utf-8")
        user, expires_ts_str, signature = decoded.split("|", 2)
        expires_ts = int(expires_ts_str)
    except (ValueError, TypeError, UnicodeDecodeError):
        return False

    if user != required_user or expires_ts <= int(datetime.now(timezone.utc).timestamp()):
        return False

    payload = f"{user}|{expires_ts}"
    expected = hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, expected)


def get_cookie_manager():
    if stx is None:
        return None
    return stx.CookieManager(key="auth_cookie_manager")


def clear_auth_cookie(cookie_manager) -> None:
    if cookie_manager is None:
        return
    try:
        cookie_manager.delete(AUTH_COOKIE_NAME, key="delete_auth_cookie")
    except Exception:
        cookie_manager.set(
            AUTH_COOKIE_NAME,
            "",
            expires_at=datetime.now() - timedelta(days=1),
            key="expire_auth_cookie",
        )


def require_auth(cookie_manager) -> None:
    if not is_auth_enabled():
        return

    required_user = os.getenv("DASHBOARD_AUTH_USER", DEFAULT_AUTH_USER)
    stored_hash = os.getenv("DASHBOARD_PASSWORD_HASH", "")
    remember_days = parse_remember_days()

    if not stored_hash:
        st.error(
            "Authentication is enabled but DASHBOARD_PASSWORD_HASH is not set. "
            "Set environment variables before starting Streamlit."
        )
        st.stop()

    if "auth_ok" not in st.session_state:
        st.session_state["auth_ok"] = False
    if "auth_failed_attempts" not in st.session_state:
        st.session_state["auth_failed_attempts"] = 0
    if "auth_lockout_until" not in st.session_state:
        st.session_state["auth_lockout_until"] = None

    auth_secret = get_auth_secret(required_user, stored_hash)

    # Query param fallback for browsers that do not persist cookies reliably on local HTTP.
    token_from_query = st.query_params.get(AUTH_QUERY_PARAM)
    if token_from_query and verify_auth_token(token_from_query, required_user, auth_secret):
        st.session_state["auth_ok"] = True
        st.session_state["auth_failed_attempts"] = 0
        st.session_state["auth_lockout_until"] = None

    if cookie_manager is not None and not st.session_state["auth_ok"]:
        # On some mobile browsers, cookie values can appear on the second rerun only.
        if "auth_cookie_probe_done" not in st.session_state:
            st.session_state["auth_cookie_probe_done"] = True
            st.rerun()

        token = cookie_manager.get(AUTH_COOKIE_NAME)
        if token and verify_auth_token(token, required_user, auth_secret):
            st.session_state["auth_ok"] = True
            st.session_state["auth_failed_attempts"] = 0
            st.session_state["auth_lockout_until"] = None
            st.session_state["auth_cookie_probe_done"] = False

    if st.session_state["auth_ok"]:
        return

    lockout_until = st.session_state["auth_lockout_until"]
    if lockout_until is not None and datetime.now() < lockout_until:
        remaining = int((lockout_until - datetime.now()).total_seconds())
        st.error(f"Too many failed attempts. Try again in {remaining}s.")
        st.stop()

    st.subheader("Secure Login")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")

    if submitted:
        valid_user = hmac.compare_digest(username.strip(), required_user)
        valid_pass = verify_password(password, stored_hash)
        if valid_user and valid_pass:
            st.session_state["auth_ok"] = True
            st.session_state["auth_failed_attempts"] = 0
            st.session_state["auth_lockout_until"] = None
            if cookie_manager is not None:
                token, expires_at = make_auth_token(required_user, auth_secret, remember_days)
                cookie_manager.set(
                    AUTH_COOKIE_NAME,
                    token,
                    expires_at=expires_at,
                    key="set_auth_cookie",
                )
                st.query_params[AUTH_QUERY_PARAM] = token
            st.session_state["auth_cookie_probe_done"] = False
            st.rerun()
        else:
            st.session_state["auth_failed_attempts"] += 1
            attempts = st.session_state["auth_failed_attempts"]
            if attempts >= 5:
                st.session_state["auth_lockout_until"] = datetime.now() + timedelta(seconds=60)
                st.session_state["auth_failed_attempts"] = 0
            st.error("Invalid username or password")
    st.stop()


def draw_sidebar_brand(nintendo_logo: str, total_logo: str) -> None:
    def load_logo(path: str):
        logo_path = Path(path)
        if not logo_path.exists():
            return None
        img = Image.open(logo_path)
        if "A" in img.getbands():
            alpha = img.getchannel("A")
            bbox = alpha.getbbox()
            if bbox is not None:
                img = img.crop(bbox)
        return img

    def image_to_data_uri(img: Image.Image) -> str:
        buf = BytesIO()
        img.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    nintendo_img = load_logo(nintendo_logo)
    total_img = load_logo(total_logo)

    nintendo_markup = "Nintendo"
    total_markup = "Total Concept"
    if nintendo_img is not None:
        nintendo_markup = f'<img src="{image_to_data_uri(nintendo_img)}" style="height:72px; width:auto;" />'
    if total_img is not None:
        total_markup = f'<img src="{image_to_data_uri(total_img)}" style="height:72px; width:auto;" />'

    st.markdown(
        f"""
        <div style="display:flex;align-items:center;justify-content:center;gap:14px;margin-bottom:10px;padding:1rem 0 1rem 0;">
            <div>{nintendo_markup}</div>
            <div style="font-size:2rem;font-weight:800;color:#e60012;line-height:1;">x</div>
            <div>{total_markup}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def normalize_database_url(url: str) -> str:
    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://") :]
    return url


def resolve_database_target(default_sqlite_path: str) -> tuple[str, str]:
    database_url = os.getenv("DATABASE_URL", "").strip()
    if database_url:
        return "postgres", normalize_database_url(database_url)
    return "sqlite", default_sqlite_path


def open_db_connection(db_backend: str, db_target: str):
    if db_backend == "postgres":
        if psycopg2 is None:
            raise RuntimeError(
                "Postgres backend selected but psycopg2 is not installed. "
                "Install psycopg2-binary and redeploy."
            )
        return psycopg2.connect(db_target)
    return sqlite3.connect(db_target)


def load_data(db_backend: str, db_target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    conn = open_db_connection(db_backend, db_target)
    sessions = pd.read_sql_query(
        """
        SELECT
            id,
            started_at,
            ended_at,
            source,
            camera_name,
            model,
            line_config_json,
            in_count,
            out_count,
            inside_count
        FROM sessions
        ORDER BY id DESC
        """,
        conn,
    )
    events = pd.read_sql_query(
        """
        SELECT
            id,
            session_id,
            event_time,
            track_id,
            direction,
            center_x,
            center_y,
            in_count,
            out_count,
            inside_count
        FROM events
        ORDER BY id DESC
        """,
        conn,
    )
    conn.close()

    if not sessions.empty:
        sessions["started_at"] = pd.to_datetime(sessions["started_at"], errors="coerce")
        sessions["ended_at"] = pd.to_datetime(sessions["ended_at"], errors="coerce")
        sessions["camera_name"] = sessions["camera_name"].fillna("")
        sessions["camera_name"] = sessions["camera_name"].replace("", "(unlabeled)")

    if not events.empty:
        events["event_time"] = pd.to_datetime(events["event_time"], errors="coerce")

    return sessions, events


def ensure_db_initialized(db_backend: str, db_target: str) -> None:
    if db_backend == "sqlite":
        db_file = Path(db_target)
        db_file.parent.mkdir(parents=True, exist_ok=True)

    conn = open_db_connection(db_backend, db_target)

    if db_backend == "postgres":
        conn.cursor().execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id BIGSERIAL PRIMARY KEY,
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
        conn.cursor().execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id BIGSERIAL PRIMARY KEY,
                session_id BIGINT NOT NULL REFERENCES sessions(id),
                event_time TEXT NOT NULL,
                track_id BIGINT NOT NULL,
                direction TEXT NOT NULL,
                center_x INTEGER NOT NULL,
                center_y INTEGER NOT NULL,
                in_count INTEGER NOT NULL,
                out_count INTEGER NOT NULL,
                inside_count INTEGER NOT NULL
            )
            """
        )
    else:
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
    conn.close()


def apply_filters(
    sessions: pd.DataFrame,
    events: pd.DataFrame,
    date_start,
    date_end,
    cameras: list[str],
    directions: list[str],
    session_ids: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered_events = events.copy()
    filtered_sessions = sessions.copy()

    if not filtered_events.empty:
        if date_start is not None:
            filtered_events = filtered_events[filtered_events["event_time"] >= pd.to_datetime(date_start)]
        if date_end is not None:
            end_dt = pd.to_datetime(date_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            filtered_events = filtered_events[filtered_events["event_time"] <= end_dt]

    if session_ids:
        filtered_events = filtered_events[filtered_events["session_id"].isin(session_ids)]
        filtered_sessions = filtered_sessions[filtered_sessions["id"].isin(session_ids)]

    if cameras and not filtered_sessions.empty:
        selected_sessions = filtered_sessions[filtered_sessions["camera_name"].isin(cameras)]
        allowed_ids = selected_sessions["id"].tolist()
        filtered_sessions = selected_sessions
        filtered_events = filtered_events[filtered_events["session_id"].isin(allowed_ids)]

    if directions:
        filtered_events = filtered_events[filtered_events["direction"].isin(directions)]

    if not filtered_events.empty:
        in_agg = (
            filtered_events[filtered_events["direction"] == "IN"]
            .groupby("session_id")
            .size()
            .rename("in_count")
        )
        out_agg = (
            filtered_events[filtered_events["direction"] == "OUT"]
            .groupby("session_id")
            .size()
            .rename("out_count")
        )
        merged = pd.concat([in_agg, out_agg], axis=1).fillna(0).astype(int)
        merged["inside_count"] = merged["in_count"] - merged["out_count"]
        merged = merged.reset_index()
        filtered_sessions = filtered_sessions.drop(columns=["in_count", "out_count", "inside_count"]).merge(
            merged,
            left_on="id",
            right_on="session_id",
            how="left",
        )
        filtered_sessions[["in_count", "out_count", "inside_count"]] = filtered_sessions[
            ["in_count", "out_count", "inside_count"]
        ].fillna(0).astype(int)
        filtered_sessions = filtered_sessions.drop(columns=["session_id"])
    else:
        filtered_sessions = filtered_sessions.copy()
        filtered_sessions["in_count"] = 0
        filtered_sessions["out_count"] = 0
        filtered_sessions["inside_count"] = 0

    return filtered_sessions, filtered_events


def build_dashboard() -> None:
    suppress_streamlit_media_logs()
    st.set_page_config(page_title="Human Counter Dashboard", layout="wide", initial_sidebar_state="expanded")
    theme = LIGHT_THEME
    apply_theme_css(theme)

    st.title("Human Counter Dashboard")
    cookie_manager = get_cookie_manager()
    require_auth(cookie_manager)

    if "db_path" not in st.session_state:
        st.session_state["db_path"] = DEFAULT_DB_PATH
    if "preview_path" not in st.session_state:
        st.session_state["preview_path"] = DEFAULT_PREVIEW_PATH

    db_path = st.session_state["db_path"]
    preview_path = st.session_state["preview_path"]

    db_backend, db_target = resolve_database_target(db_path)
    try:
        ensure_db_initialized(db_backend, db_target)
    except Exception as exc:
        st.error(f"Unable to initialize database ({db_backend}) at {db_target}: {exc}")
        st.stop()

    sessions, events = load_data(db_backend, db_target)

    with st.sidebar:
        draw_sidebar_brand(DEFAULT_NINTENDO_LOGO, DEFAULT_TOTAL_CONCEPT_LOGO)

        with st.expander("Filters", expanded=False):
            min_date = None
            max_date = None
            if not events.empty and events["event_time"].notna().any():
                min_date = events["event_time"].min().date()
                max_date = events["event_time"].max().date()

            date_start = st.date_input("Start date", value=min_date if min_date else None)
            date_end = st.date_input("End date", value=max_date if max_date else None)

            all_cameras = sorted(sessions["camera_name"].dropna().unique().tolist()) if not sessions.empty else []
            selected_cameras = st.multiselect("Camera", options=all_cameras, default=all_cameras)

            all_dirs = ["IN", "OUT"]
            selected_dirs = st.multiselect("Direction", options=all_dirs, default=all_dirs)

            all_session_ids = (
                sessions["id"].astype(int).sort_values(ascending=False).tolist() if not sessions.empty else []
            )
            selected_sessions = st.multiselect("Session IDs", options=all_session_ids, default=[])

        with st.expander("Realtime", expanded=False):
            preview_enabled = st.toggle("Enable preview module", value=False)
            data_auto_refresh = st.toggle("Auto refresh data", value=True)
            data_refresh_seconds = st.slider(
                "Data refresh interval (seconds)", min_value=0.5, max_value=30.0, value=2.0, step=0.5
            )
            preview_auto_refresh = st.toggle("Auto refresh preview", value=False, disabled=not preview_enabled)
            preview_refresh_seconds = st.slider(
                "Preview refresh interval (seconds)",
                min_value=0.2,
                max_value=5.0,
                value=0.6,
                step=0.2,
                disabled=not preview_enabled,
            )

        with st.expander("Data Source", expanded=False):
            if db_backend == "postgres":
                st.info("Database source: Postgres via DATABASE_URL")
                new_db_path = db_path
            else:
                new_db_path = st.text_input("SQLite DB path", value=db_path)
            new_preview_path = st.text_input("Preview image path", value=preview_path, disabled=not preview_enabled)
            manual_refresh = st.button("Manual refresh")

        if new_db_path != st.session_state["db_path"]:
            st.session_state["db_path"] = new_db_path
            st.rerun()
        if new_preview_path != st.session_state["preview_path"]:
            st.session_state["preview_path"] = new_preview_path
            st.rerun()
        if manual_refresh:
            st.rerun()

        active_intervals = []
        if data_auto_refresh:
            active_intervals.append(data_refresh_seconds)
        if preview_enabled and preview_auto_refresh:
            active_intervals.append(preview_refresh_seconds)

        if active_intervals:
            refresh_seconds = min(active_intervals)
            refresh_ms = int(refresh_seconds * 1000)
            st_autorefresh(interval=refresh_ms, key=f"dashboard_auto_refresh_{refresh_ms}")

        if is_auth_enabled():
            st.markdown('<div id="sidebar-logout-anchor"></div>', unsafe_allow_html=True)
            if st.button("Logout", key="logout_btn", type="primary"):
                st.session_state["auth_ok"] = False
                clear_auth_cookie(cookie_manager)
                st.query_params.pop(AUTH_QUERY_PARAM, None)
                st.rerun()

    if preview_enabled:
        st.subheader("Live Preview")
        preview_file = Path(preview_path)
        latest_preview = resolve_latest_preview(preview_file)
        if latest_preview is not None:
            updated = datetime.fromtimestamp(latest_preview.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            image_bytes = read_image_bytes(latest_preview)
            if image_bytes is not None:
                st.image(image_bytes, width="stretch")
            else:
                st.info("Preview image is temporarily locked. Waiting for next update...")
            st.caption(f"Preview source: {latest_preview} | Last update: {updated}")
        else:
            st.info(
                "Preview image not found yet. Start human_counter.py and ensure --preview-frame-path matches this path."
            )

    if sessions.empty:
        st.warning("No sessions found in database yet. Run the counter app and generate some events first.")
        st.stop()

    filtered_sessions, filtered_events = apply_filters(
        sessions,
        events,
        date_start,
        date_end,
        selected_cameras,
        selected_dirs,
        selected_sessions,
    )

    total_in = int(filtered_events[filtered_events["direction"] == "IN"].shape[0])
    total_out = int(filtered_events[filtered_events["direction"] == "OUT"].shape[0])

    current_inside = 0
    if not filtered_sessions.empty:
        active_sessions = filtered_sessions[filtered_sessions["ended_at"].isna()]
        if not active_sessions.empty:
            current_inside = int(active_sessions.sort_values("id", ascending=False)["inside_count"].iloc[0])
        else:
            current_inside = int(filtered_sessions.sort_values("id", ascending=False)["inside_count"].iloc[0])

    # Guard against tracker noise producing temporary negative occupancy values.
    current_inside = max(0, current_inside)

    today_date = datetime.now().date()
    if not filtered_events.empty and filtered_events["event_time"].notna().any():
        total_visitors_today = int(
            filtered_events[
                (filtered_events["direction"] == "IN")
                & (filtered_events["event_time"].dt.date == today_date)
            ].shape[0]
        )
    else:
        total_visitors_today = 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Currently Inside Store", current_inside)
    m2.metric("Total Visitors Today", total_visitors_today)
    m3.metric("Total Entries", total_in)
    m4.metric("Total Exits", total_out)

    color_map = {"IN": theme["in_color"], "OUT": theme["out_color"]}

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Daily Traffic")
        if filtered_events.empty:
            st.info("No events match current filters.")
        else:
            daily = filtered_events.copy()
            daily["date"] = daily["event_time"].dt.date
            daily = daily.groupby(["date", "direction"]).size().reset_index(name="count")
            fig = px.bar(
                daily,
                x="date",
                y="count",
                color="direction",
                barmode="group",
                labels={"date": "Date", "count": "Count"},
                color_discrete_map=color_map,
                template=theme["template"],
            )
            st.plotly_chart(fig, width="stretch")

    with c2:
        st.subheader("Hourly Trend")
        if filtered_events.empty:
            st.info("No events match current filters.")
        else:
            hourly = filtered_events.copy()
            hourly["hour"] = hourly["event_time"].dt.hour
            hourly = hourly.groupby(["hour", "direction"]).size().reset_index(name="count")
            fig = px.line(
                hourly,
                x="hour",
                y="count",
                color="direction",
                markers=True,
                labels={"hour": "Hour of Day", "count": "Count"},
                color_discrete_map=color_map,
                template=theme["template"],
            )
            fig.update_xaxes(dtick=1)
            st.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Occupancy Over Time")
        if filtered_events.empty:
            st.info("No events match current filters.")
        else:
            occupancy = filtered_events.sort_values("event_time")[
                ["event_time", "inside_count", "session_id"]
            ]
            fig = px.line(
                occupancy,
                x="event_time",
                y="inside_count",
                color="session_id",
                labels={"event_time": "Event Time", "inside_count": "Inside", "session_id": "Session"},
                template=theme["template"],
            )
            fig.update_traces(line_shape="hv")
            st.plotly_chart(fig, width="stretch")

    with c4:
        st.subheader("Direction Split")
        if filtered_events.empty:
            st.info("No events match current filters.")
        else:
            split = filtered_events.groupby("direction").size().reset_index(name="count")
            fig = px.pie(
                split,
                names="direction",
                values="count",
                hole=0.45,
                color="direction",
                color_discrete_map=color_map,
                template=theme["template"],
            )
            st.plotly_chart(fig, width="stretch")

    st.subheader("Session Summary")
    session_cols = [
        "id",
        "started_at",
        "ended_at",
        "camera_name",
        "source",
        "model",
        "in_count",
        "out_count",
        "inside_count",
    ]
    st.dataframe(filtered_sessions[session_cols], width="stretch", hide_index=True)

    st.subheader("Recent Events")
    event_cols = [
        "id",
        "session_id",
        "event_time",
        "track_id",
        "direction",
        "center_x",
        "center_y",
        "in_count",
        "out_count",
        "inside_count",
    ]
    st.dataframe(
        filtered_events.sort_values("id", ascending=False)[event_cols],
        width="stretch",
        hide_index=True,
    )


if __name__ == "__main__":
    build_dashboard()
