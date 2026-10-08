"""
Streamlit web UI for the vehicle + license plate logger.

    streamlit run app.py

- Lines tab:     click on a video frame to place the START and END lines (defaults to lines.json)
- Detect tab:    run the logger with a live preview, then explore counts, charts and the plate log
- Sidebar:       pick the video (defaults to vehicle-plate.mp4) and tune model / OCR
"""
import base64
import io
import json
import subprocess
import time
import zipfile
from datetime import datetime
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from ultralytics import YOLO

from detect_plate import (END_COLOR, START_COLOR, Logger, PlateCounter, PlateReader, clean_text,
                          crossed, fix_plate, parse_source)

ROOT = Path(__file__).parent
DEFAULT_VIDEO = ROOT / "vehicle-plate.mp4"
DEFAULT_LINES = ROOT / "lines.json"
UPLOAD_DIR = ROOT / "output" / "uploads"
RUNS_DIR = ROOT / "output" / "runs"
PREVIEW_W = 960      # live preview width
EDITOR_W = 1280      # line editor image width

# Chart colors (validated categorical slots + fixed status palette)
C_FRAME = "#2a78d6"  # vehicles in frame
C_ACTIVE = "#eb6834" # vehicles between the lines
C_MUTED = "#9a9893"
STATUS = ["OK", "REVIEW", "NO_PLATE"]
STATUS_COLOR = ["#0ca30c", "#fab219", C_MUTED]
STATUS_LABEL = {"OK": "✅ OK", "REVIEW": "⚠️ REVIEW", "NO_PLATE": "➖ NO_PLATE"}
STEPS = ["START point 1", "START point 2", "END point 1", "END point 2"]

st.set_page_config(page_title="Plate Logger", page_icon="🚗", layout="wide")


# ---------------------------------------------------------------- helpers
@st.cache_data(show_spinner=False)
def probe(source: str):
    src, is_live = parse_source(source)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 1 or fps > 120:
        fps = 25.0
    info = dict(fps=fps, w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                frames=0 if is_live else int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), is_live=is_live)
    info["duration"] = info["frames"] / fps if info["frames"] else 0.0
    cap.release()
    return info


@st.cache_data(show_spinner=False, ttl=60, max_entries=64)
def grab_frame(source: str, t: float):
    """RGB frame at t seconds (video files) or a fresh frame (live sources)."""
    src, is_live = parse_source(source)
    cap = cv2.VideoCapture(src)
    frame = None
    if is_live:
        for _ in range(10):  # webcams start dark
            ok, f = cap.read()
            frame = f if ok else frame
    else:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(t, 0) * 1000)
        ok, frame = cap.read()
        frame = frame if ok else None
    cap.release()
    return None if frame is None else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


@st.cache_resource(show_spinner="Loading model...")
def model_names(model_path: str) -> dict:
    return YOLO(model_path).names


@st.cache_resource(show_spinner="Loading OCR (the first run downloads the EasyOCR model)...")
def ocr_reader():
    import torch
    return PlateReader(gpu=torch.cuda.is_available())


def resize_w(img, width):
    h, w = img.shape[:2]
    if w <= width:
        return img
    return cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def to_browser_mp4(raw: Path, out: Path) -> bool:
    """OpenCV writes mp4v, which browsers can't play; re-encode to H.264."""
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(raw), "-vf", "scale='min(1280,iw)':-2",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(out)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception:
        return False


def load_default_lines():
    if DEFAULT_LINES.exists():
        d = json.loads(DEFAULT_LINES.read_text())
        return [list(p) for p in d["start"] + d["end"]]
    return []


def lines_json(pts):
    return json.dumps({"start": pts[:2], "end": pts[2:]}, indent=2)


def thumb_uri(path, height=48):
    """Small JPEG as a data URI, for image columns in tables."""
    if not isinstance(path, str) or not Path(path).exists():
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    img = cv2.resize(img, (max(1, int(w * height / h)), height), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else None


def read_rgb(path, width=480):
    img = cv2.imread(path) if isinstance(path, str) and Path(path).exists() else None
    return None if img is None else resize_w(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width)


# ---------------------------------------------------------------- sidebar: source
st.sidebar.title("🚗 Plate logger")
st.sidebar.subheader("Video source")
src_mode = st.sidebar.radio("Source", [f"Default ({DEFAULT_VIDEO.name})", "Upload video", "Path / RTSP URL / webcam"],
                            label_visibility="collapsed")
source = None
if src_mode.startswith("Default"):
    source = str(DEFAULT_VIDEO)
elif src_mode == "Upload video":
    up = st.sidebar.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv", "webm", "m4v"])
    if up is not None:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        dest = UPLOAD_DIR / Path(up.name).name
        if not dest.exists() or dest.stat().st_size != up.size:
            dest.write_bytes(up.getbuffer())
        source = str(dest)
else:
    typed = st.sidebar.text_input("Path, URL or webcam index", placeholder="0  or  rtsp://user:pass@ip/stream",
                                  help="Webcams and local cameras only work when the app runs on your own machine")
    source = typed.strip() or None

info = probe(source) if source else None
if source and info is None:
    st.sidebar.error(f"Could not open: {source}")
elif info:
    st.sidebar.caption(
        f"{info['w']}×{info['h']} · {info['fps']:.0f} fps"
        + ("" if info["is_live"] else f" · {info['duration']:.1f}s · {info['frames']} frames")
    )

# Lines are stored as fractions of the frame (like lines.json), so they survive a video change.
ss = st.session_state
ss.setdefault("pts", load_default_lines())
ss.setdefault("click_key", 0)
ss.setdefault("last_click", None)

# ---------------------------------------------------------------- sidebar: settings
st.sidebar.subheader("Model")
pt_files = sorted(p.name for p in ROOT.glob("*.pt")) or ["best.pt"]
model_path = st.sidebar.selectbox("Weights", pt_files, index=pt_files.index("best.pt") if "best.pt" in pt_files else 0)
names = model_names(model_path)
plate_classes = [v for v in names.values() if "plate" in v.lower()]
st.sidebar.caption(f"Plate class: **{', '.join(plate_classes) or 'none'}** · vehicles: "
                   + ", ".join(v for v in names.values() if v not in plate_classes))
conf = st.sidebar.slider("Detection confidence", 0.05, 0.95, 0.25, 0.05)
imgsz = st.sidebar.select_slider("Inference size", [320, 480, 640, 800, 960, 1280], value=640,
                                 help="Larger finds small/distant plates but is slower")
tracker = st.sidebar.selectbox("Tracker", ["bytetrack.yaml", "botsort.yaml"])

st.sidebar.subheader("Plate reading")
use_ocr = st.sidebar.toggle("Read plates (OCR)", value=bool(plate_classes), disabled=not plate_classes,
                            help="Off = only count vehicles and test the lines (much faster)")
id_format = st.sidebar.toggle("Force Indonesian format", value=False,
                              help="Fix O/0, B/8 ... to fit 'B 1234 XYZ'. Off = store the text as read")
top_k = st.sidebar.number_input("Plate crops read per vehicle", 1, 15, 5,
                                help="The sharpest crops between the lines are read and voted")
min_ocr_conf = st.sidebar.slider("Min OCR confidence", 0.0, 1.0, 0.25, 0.05,
                                 help="Below this, the plate status becomes REVIEW")

st.sidebar.subheader("Performance")
stride = st.sidebar.number_input("Process every Nth frame", 1, 10, 1,
                                 help="2 = skip every other frame (≈2× faster), but small fast-moving plates "
                                      "then lose their track and many vehicles end up NO_PLATE")
save_video = st.sidebar.checkbox("Save annotated video", value=True)

if not info:
    st.info("Choose a video source in the sidebar to get started.")
    st.stop()

W, H = info["w"], info["h"]
pts = ss.pts
px = [(int(x * W), int(y * H)) for x, y in pts]
crossing = len(px) == 4 and crossed(px[0], px[1], px[2], px[3])
lines_ok = len(px) == 4 and px[0] != px[1] and px[2] != px[3] and not crossing

tab_lines, tab_detect = st.tabs(["① Lines", "② Detect & dashboard"])

# ---------------------------------------------------------------- line editor
with tab_lines:
    left, right = st.columns([3, 1], gap="large")
    with right:
        st.markdown("**Place the counting lines**")
        st.caption("Click 4 points: START (2 points), then END (2 points). "
                   "Vehicles are logged once when they cross START, then END. Reverse traffic is ignored.")
        if len(pts) < 4:
            st.info(f"Next click: **{STEPS[len(pts)]}**")
        t = 0.0
        if not info["is_live"] and info["duration"] > 0:
            t = st.slider("Frame at (s)", 0.0, max(info["duration"] - 0.1, 0.0), min(5.0, info["duration"] / 2), 0.5)
        elif st.button("Grab new frame", width="stretch"):
            grab_frame.clear()

        def _reset_clicks():
            ss.click_key += 1
            ss.last_click = None

        c1, c2 = st.columns(2)
        if c1.button("↶ Undo", width="stretch", disabled=not pts):
            pts.pop()
            st.rerun()
        if c2.button("✕ Clear", width="stretch", disabled=not pts):
            pts.clear()
            _reset_clicks()
            st.rerun()
        c3, c4 = st.columns(2)
        if c3.button("⇅ Swap", width="stretch", disabled=len(pts) != 4, help="Reverse the direction (swap START and END)"):
            ss.pts = pts[2:] + pts[:2]
            st.rerun()
        if c4.button("Reset", width="stretch", help="Reload lines.json", disabled=not DEFAULT_LINES.exists()):
            ss.pts = load_default_lines()
            _reset_clicks()
            st.rerun()
        if st.button("💾 Save to lines.json", width="stretch", type="primary", disabled=not lines_ok,
                     help="Also used by detect_plate.py. On Streamlit Cloud this only lasts until the app restarts; "
                          "download the JSON to keep it."):
            DEFAULT_LINES.write_text(lines_json(pts))
            st.toast("Saved lines.json")

        if crossing:
            st.warning("The START and END lines cross each other. Undo and place them apart.")
        elif len(pts) == 4 and not lines_ok:
            st.warning("A line has zero length. Undo and click two different points.")
        elif lines_ok:
            st.success("Lines ready · green = START, red = END")

        st.download_button("Download lines JSON", lines_json(pts), "lines.json", "application/json",
                           width="stretch", disabled=not lines_ok)

        def _load_lines_upload():
            f = ss.get("lines_upload")
            if f is None:
                return
            try:
                d = json.loads(f.getvalue())
                new = [[float(x), float(y)] for x, y in d["start"] + d["end"]]
                if len(new) != 4 or not all(0 <= v <= 1 for p in new for v in p):
                    raise ValueError("expected 2 START + 2 END points as fractions (0-1) of the frame")
                ss.pts = new
                _reset_clicks()
            except Exception as e:
                ss.lines_upload_error = f"Invalid lines file: {e}"

        st.file_uploader("Load lines JSON", type=["json"], key="lines_upload", on_change=_load_lines_upload)
        if err := ss.pop("lines_upload_error", None):
            st.error(err)

        with st.expander("Points (pixels)"):
            st.dataframe(pd.DataFrame(px, columns=["x", "y"], index=STEPS[:len(px)]) if px
                         else pd.DataFrame(columns=["x", "y"]), height=200)

    with left:
        frame = grab_frame(source, t)
        if frame is None:
            st.error("Could not read a frame from this source.")
        else:
            img = resize_w(frame, EDITOR_W).copy()
            s = img.shape[1] / W
            p = [(int(x * s), int(y * s)) for x, y in px]
            rgb = lambda bgr: bgr[::-1]
            for i, (a, b, col, name) in enumerate([(0, 1, START_COLOR, "START"), (2, 3, END_COLOR, "END")]):
                if len(p) > b:
                    cv2.line(img, p[a], p[b], rgb(col), 3)
                if len(p) > a:
                    cv2.putText(img, name, (p[a][0] + 8, p[a][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
                    cv2.putText(img, name, (p[a][0] + 8, p[a][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, rgb(col), 2)
            for i, q in enumerate(p):
                cv2.circle(img, q, 7, rgb(START_COLOR if i < 2 else END_COLOR), -1)
                cv2.circle(img, q, 7, (255, 255, 255), 2)
            if len(p) == 4:  # travel direction: START midpoint -> END midpoint
                m1 = ((p[0][0] + p[1][0]) // 2, (p[0][1] + p[1][1]) // 2)
                m2 = ((p[2][0] + p[3][0]) // 2, (p[2][1] + p[3][1]) // 2)
                cv2.arrowedLine(img, m1, m2, (255, 255, 0), 3, tipLength=0.04)

            click = streamlit_image_coordinates(img, width="stretch", key=f"line_click_{ss.click_key}",
                                                cursor="crosshair", image_format="JPEG", jpeg_quality=85)
            if click and click.get("unix_time") != ss.last_click:
                ss.last_click = click.get("unix_time")
                if len(pts) < 4:
                    x = min(max(click["x"] / click["width"], 0.0), 1.0)
                    y = min(max(click["y"] / click["height"], 0.0), 1.0)
                    pts.append([round(x, 4), round(y, 4)])
                    st.rerun()
                else:
                    st.toast("Both lines are set. Undo or Clear to move them.")


# ---------------------------------------------------------------- charts
def axis_cfg(chart):
    return (chart.configure_axis(grid=True, gridColor="#8883", domain=False, tickColor="#8886",
                                 labelColor=C_MUTED, titleColor=C_MUTED, labelFontSize=11, titleFontSize=11)
            .configure_view(stroke=None)
            .configure_legend(orient="top", labelColor=C_MUTED, title=None, symbolStrokeWidth=3))


def traffic_chart(tl: pd.DataFrame, rows: pd.DataFrame, height=260):
    """Vehicles in frame / between the lines over time, with a tick for each logged vehicle."""
    long = tl.melt("t", ["in_frame", "between"], var_name="series", value_name="vehicles")
    long["series"] = long["series"].map({"in_frame": "In frame", "between": "Between lines"})
    color = alt.Color("series:N", scale=alt.Scale(domain=["In frame", "Between lines"], range=[C_FRAME, C_ACTIVE]))
    x = alt.X("t:Q", title="Video time (s)")

    hover = alt.selection_point(fields=["t"], nearest=True, on="pointerover", empty=False, clear="pointerout")
    lines = alt.Chart(long).mark_line(strokeWidth=2, interpolate="step-after").encode(
        x=x, y=alt.Y("vehicles:Q", title="Vehicles", axis=alt.Axis(tickMinStep=1)), color=color)
    points = lines.mark_point(size=60, filled=True, opacity=1).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)))
    rule = alt.Chart(tl).mark_rule(color=C_MUTED).encode(
        x="t:Q",
        opacity=alt.condition(hover, alt.value(0.6), alt.value(0)),
        tooltip=[alt.Tooltip("t:Q", title="Time (s)", format=".1f"),
                 alt.Tooltip("in_frame:Q", title="In frame"),
                 alt.Tooltip("between:Q", title="Between lines"),
                 alt.Tooltip("logged:Q", title="Logged so far")],
    ).add_params(hover)
    layers = [lines, points, rule]
    if len(rows):
        ev = rows.assign(label=rows["plate"].fillna("–"))
        ticks = alt.Chart(ev).mark_tick(thickness=2, size=14, opacity=0.9).encode(
            x="video_time_s:Q", y=alt.value(height - 7),
            color=alt.Color("plate_status:N", scale=alt.Scale(domain=STATUS, range=STATUS_COLOR), legend=None),
            tooltip=[alt.Tooltip("video_time_s:Q", title="Logged at (s)", format=".1f"),
                     alt.Tooltip("vehicle_type:N", title="Type"), alt.Tooltip("label:N", title="Plate"),
                     alt.Tooltip("plate_status:N", title="Status")])
        layers.append(ticks)
    return axis_cfg(alt.layer(*layers).properties(height=height))


def type_chart(rows: pd.DataFrame):
    """Vehicles per type, split by plate status."""
    df = rows.groupby(["vehicle_type", "plate_status"]).size().reset_index(name="n")
    order = rows["vehicle_type"].value_counts().index.tolist()
    chart = alt.Chart(df).mark_bar(cornerRadiusEnd=4, height={"band": 0.7}).encode(
        y=alt.Y("vehicle_type:N", sort=order, title=None),
        x=alt.X("n:Q", title="Vehicles", axis=alt.Axis(tickMinStep=1, format="d")),
        color=alt.Color("plate_status:N", scale=alt.Scale(domain=STATUS, range=STATUS_COLOR)),
        order=alt.Order("plate_status:N", sort="ascending"),
        tooltip=[alt.Tooltip("vehicle_type:N", title="Type"), alt.Tooltip("plate_status:N", title="Status"),
                 alt.Tooltip("n:Q", title="Vehicles")],
    )
    return axis_cfg(chart.properties(height=max(120, 44 * len(order))))


def ocr_conf_chart(rows: pd.DataFrame, threshold: float):
    df = rows[rows["plate_status"] != "NO_PLATE"]
    bars = alt.Chart(df).mark_bar(color=C_FRAME, cornerRadiusEnd=3).encode(
        x=alt.X("plate_conf:Q", bin=alt.Bin(step=0.05, extent=[0, 1]), title="OCR confidence (mean of readings)"),
        y=alt.Y("count():Q", title="Vehicles", axis=alt.Axis(tickMinStep=1, format="d")),
        tooltip=[alt.Tooltip("plate_conf:Q", bin=alt.Bin(step=0.05, extent=[0, 1]), title="Confidence"),
                 alt.Tooltip("count():Q", title="Vehicles")],
    )
    rule = alt.Chart(pd.DataFrame({"x": [threshold]})).mark_rule(color=C_MUTED, strokeDash=[4, 3]).encode(x="x:Q")
    return axis_cfg((bars + rule).properties(height=240))


# ---------------------------------------------------------------- stats
ROW_COLS = ["id", "logged_at", "source", "track_id", "vehicle_type", "plate", "plate_status", "plate_conf",
            "plate_agree", "plate_reads", "completed", "video_time_s", "snapshot", "plate_image"]


def rows_df(run) -> pd.DataFrame:
    return pd.DataFrame(list(run["logger"].rows), columns=ROW_COLS)


def render_kpis(run):
    rows = rows_df(run)
    n = len(rows)
    st_counts = rows["plate_status"].value_counts()
    tl = pd.DataFrame(run["timeline"])
    elapsed = run["wall_elapsed"] or 1e-9
    k = st.columns(6)
    k[0].metric("🚗 Vehicles logged", n, border=True,
                chart_data=tl["logged"].tolist()[-200:] if len(tl) else None, chart_type="area")
    k[1].metric("✅ Plate OK", int(st_counts.get("OK", 0)), border=True,
                help="Readings agree and confidence is above the minimum")
    k[2].metric("⚠️ Review", int(st_counts.get("REVIEW", 0)), border=True,
                help="Unreadable, or the readings disagree. Check the plate image.")
    k[3].metric("➖ No plate", int(st_counts.get("NO_PLATE", 0)), border=True,
                help="No plate was seen between the lines")
    k[4].metric("Read rate", f"{st_counts.get('OK', 0) / n * 100:.0f}%" if n else "–", border=True,
                help="Share of logged vehicles with an OK plate")
    k[5].metric("Processing speed", f"{run['processed'] / elapsed:.1f} fps", border=True,
                help=f"{run['processed']} frames analysed in {elapsed:.0f}s")


def snapshots_zip(rows: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("vehicles.csv", rows.to_csv(index=False))
        for col in ("snapshot", "plate_image"):
            for p in rows[col].dropna():
                if Path(p).exists():
                    z.write(p, f"snapshots/{Path(p).name}")
    return buf.getvalue()


def render_dashboard(run):
    status = {"finished": "✅ Finished", "stopped": "⏹ Stopped early", "running": "⏳ Running",
              "error": "❌ Error"}[run["status"]]
    st.markdown(f"#### Results · {status}")
    s = run["settings"]
    st.caption(f"{run['source']} · model {s['model']} · conf {s['conf']} · OCR {'on' if s['ocr'] else 'off'}"
               f"{' (Indonesian format)' if s['id_format'] else ''} · every {s['stride']} frame(s) · "
               f"saved to `{run['dir'].relative_to(ROOT).as_posix()}`")
    if run.get("error"):
        st.error(run["error"])
    render_kpis(run)

    rows = rows_df(run)
    tl = pd.DataFrame(run["timeline"])

    if len(tl):
        st.markdown("**Traffic over time** · ticks along the bottom = vehicles logged (colored by plate status)")
        st.altair_chart(traffic_chart(tl, rows), width="stretch")

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("**Vehicles per type** · by plate status")
        if len(rows):
            st.altair_chart(type_chart(rows), width="stretch")
        else:
            st.info("No vehicle crossed START → END in this run.")
    with b:
        st.markdown("**Plate OCR confidence** · dashed line = min OCR confidence")
        if (rows["plate_status"] != "NO_PLATE").any() and s["ocr"]:
            st.altair_chart(ocr_conf_chart(rows, s["min_ocr_conf"]), width="stretch")
            st.caption("Many plates just below the line? Lower the minimum or raise the crops per vehicle.")
        else:
            st.info("No plates were read in this run.")

    st.markdown("**Vehicle log** · edit the *Plate* column to correct a reading; downloads include your edits")
    if len(rows):
        table = rows.assign(
            plate_crop=[thumb_uri(p) for p in rows["plate_image"]],
            status=rows["plate_status"].map(STATUS_LABEL),
            completed=rows["completed"].astype(bool),
        )[["id", "video_time_s", "track_id", "vehicle_type", "plate_crop", "plate", "status", "plate_conf",
           "plate_agree", "completed", "plate_reads"]]
        edited = st.data_editor(
            table, hide_index=True, width="stretch", key=f"log_{run['dir'].name}",
            disabled=[c for c in table.columns if c != "plate"],
            column_config={
                "id": st.column_config.NumberColumn("#", width="small"),
                "video_time_s": st.column_config.NumberColumn("Video time (s)", format="%.1f"),
                "track_id": st.column_config.NumberColumn("Track"),
                "vehicle_type": "Type",
                "plate_crop": st.column_config.ImageColumn("Plate image", width="medium"),
                "plate": st.column_config.TextColumn("Plate ✏️"),
                "status": "Status",
                "plate_conf": st.column_config.ProgressColumn("OCR conf", min_value=0, max_value=1, format="%.2f"),
                "plate_agree": st.column_config.ProgressColumn("Agreement", min_value=0, max_value=1, format="%.2f",
                                                               help="Share of readings that match the final plate"),
                "completed": st.column_config.CheckboxColumn("Reached END",
                                                             help="Off = crossed START but was lost before END"),
                "plate_reads": st.column_config.TextColumn("All readings", width="large"),
            })
        rows = rows.assign(plate=edited["plate"].values)

        review = rows[rows["plate_status"] == "REVIEW"]
        with st.expander(f"⚠️ Review queue ({len(review)})", expanded=bool(len(review))):
            if not len(review):
                st.caption("Nothing to review.")
            for r in review.itertuples():
                c = st.columns([2, 2, 3])
                if (im := read_rgb(r.snapshot, 480)) is not None:
                    c[0].image(im, caption=f"#{r.id} · {r.vehicle_type} · {r.video_time_s:.1f}s")
                if (im := read_rgb(r.plate_image, 360)) is not None:
                    c[1].image(im, caption="Best plate crop")
                c[2].markdown(f"**Plate:** `{r.plate or '?'}` · conf {r.plate_conf:.2f} · agreement {r.plate_agree:.2f}")
                c[2].caption("Readings: " + (r.plate_reads or "–"))

        with st.expander(f"Snapshots ({len(rows)})"):
            cols = st.columns(4)
            for i, r in enumerate(rows.itertuples()):
                if (im := read_rgb(r.snapshot, 480)) is not None:
                    cols[i % 4].image(im, caption=f"#{r.id} · {r.vehicle_type} · {r.plate or '–'} · {r.plate_status}")
    else:
        st.caption("No vehicles logged.")

    st.markdown("**Downloads**")
    d = st.columns(4)
    d[0].download_button("Vehicle log (CSV)", rows.to_csv(index=False), "vehicles.csv", "text/csv",
                         width="stretch", disabled=not len(rows))
    if len(rows):
        d[1].download_button("Log + snapshots (ZIP)", snapshots_zip(rows), f"plates_{run['dir'].name}.zip",
                             "application/zip", width="stretch")
    if len(tl):
        d[2].download_button("Timeline (CSV)", tl.to_csv(index=False), "timeline.csv", "text/csv", width="stretch")

    if run.get("video_raw") and run["status"] != "running":
        mp4 = run["dir"] / "output.mp4"
        if not mp4.exists():
            with st.spinner("Encoding annotated video for the browser..."):
                if not to_browser_mp4(run["video_raw"], mp4):
                    mp4 = None
        if mp4 and mp4.exists():
            d[3].download_button("Annotated video", mp4.read_bytes(), "output.mp4", "video/mp4", width="stretch")
            st.markdown("**Annotated video**")
            st.video(str(mp4))
        else:
            st.warning(f"Couldn't re-encode the video for playback. Raw file: {run['video_raw']}")


# ---------------------------------------------------------------- detection run
def run_detection(live_img, live_stats, progress):
    run_dir = RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    reader = ocr_reader() if use_ocr else None
    logger = Logger(str(run_dir / "vehicles.db"), str(run_dir), reader, min_ocr_conf,
                    fix_plate if id_format else clean_text)
    logger.start()
    run = dict(
        dir=run_dir, source=Path(source).name if not info["is_live"] else source, logger=logger,
        settings=dict(model=model_path, conf=conf, imgsz=imgsz, stride=stride, ocr=reader is not None,
                      id_format=id_format, min_ocr_conf=min_ocr_conf),
        status="running", error=None, timeline=[], processed=0,
        wall_start=time.time(), wall_elapsed=0.0, video_raw=None,
    )
    ss.run = run
    model = YOLO(model_path)  # fresh model = fresh tracker state
    counter = PlateCounter(model, px[:2], px[2:], logger, run["source"], conf=conf, imgsz=imgsz,
                           tracker=tracker, top_k=top_k)
    src, is_live = parse_source(source)
    cap = cv2.VideoCapture(src)
    fps, total = info["fps"], info["frames"]
    writer = None
    if save_video:
        run["video_raw"] = run_dir / "output_raw.mp4"
        writer = cv2.VideoWriter(str(run["video_raw"]), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (W, H))

    frame_idx, last_ui = 0, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if is_live:
                    time.sleep(1)
                    cap.release()
                    cap = cv2.VideoCapture(src)
                    continue
                break
            frame_idx += 1
            if (frame_idx - 1) % stride:
                continue
            now = (time.time() - run["wall_start"]) if is_live else (frame_idx - 1) / fps
            counter.update(frame, now)
            run["timeline"].append(dict(t=round(now, 2), in_frame=len(counter.vehicles), between=counter.active,
                                        logged=sum(counter.counts.values())))
            run["processed"] += 1
            run["wall_elapsed"] = time.time() - run["wall_start"]

            vis = counter.draw(frame)
            if writer:
                writer.write(vis)
            live_img.image(cv2.cvtColor(resize_w(vis, PREVIEW_W), cv2.COLOR_BGR2RGB), width="stretch")
            if total:
                progress.progress(min(frame_idx / total, 1.0), text=f"{now:.1f}s / {info['duration']:.1f}s")
            if time.time() - last_ui > 1.5:
                last_ui = time.time()
                with live_stats.container():
                    render_kpis(run)
                    rows = rows_df(run)
                    if len(rows):
                        st.dataframe(rows[["id", "vehicle_type", "plate", "plate_status"]].iloc[::-1].head(8),
                                     hide_index=True, width="stretch")
        counter.finish()
        run["status"] = "finished"
    except Exception as e:
        run["status"], run["error"] = "error", f"{type(e).__name__}: {e}"
    finally:
        # Also runs when Streamlit interrupts the script (Stop button / any widget change)
        if run["status"] == "running":
            run["status"] = "stopped"
        cap.release()
        if writer:
            writer.release()
        logger.jobs.put(None)  # let OCR finish the vehicles already handed over
        logger.join()
        run["wall_elapsed"] = time.time() - run["wall_start"]


with tab_detect:
    run = ss.get("run")
    c1, c2, c3 = st.columns([1, 1, 4])
    start = c1.button("▶ Start", type="primary", width="stretch", disabled=not lines_ok)
    c2.button("⏹ Stop", width="stretch", help="Stops the run; vehicles logged so far are kept")
    if not lines_ok:
        c3.warning("Place valid START and END lines in the Lines tab first.")
    else:
        c3.caption(f"Model **{model_path}** on **{Path(source).name if not info['is_live'] else source}** · "
                   f"OCR {'on' if use_ocr else 'off'}. Changing any setting while running stops the run.")

    if start:
        progress = st.empty() if info["is_live"] else st.progress(0.0, text="Starting...")
        v, s_ = st.columns([3, 2], gap="large")
        live_img = v.empty()
        live_stats = s_.empty()
        with st.spinner("Running..."):
            run_detection(live_img, live_stats, progress)
        st.rerun()  # redraw the page with the full dashboard
    elif run:
        render_dashboard(run)
    else:
        st.info("Press **Start** to analyse the video. A live preview and the latest plates appear here while it runs.")
