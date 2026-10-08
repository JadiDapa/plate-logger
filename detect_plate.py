"""
Vehicle + license plate logger (prototype for the bag-inspection line).

How it works
  - Every vehicle gets a tracking ID (YOLO + ByteTrack).
  - It must cross the START line, then the END line (one direction only).
  - Between the two lines, the sharpest plate crops are kept.
  - At the END line the crops are read with OCR, the readings are voted into one
    plate number, and ONE row is saved to SQLite with: id, vehicle type, plate,
    status, time and snapshots. Counts per vehicle type are shown live.

Usage
  1) Draw the two lines once per camera view (click 4 points, press S):
       python detect_plate.py --source traffic.mp4 --setup
  2) Run:
       python detect_plate.py --source traffic.mp4 --model best.pt
       python detect_plate.py --source 0                         # webcam
       python detect_plate.py --source "rtsp://user:pass@ip/stream"
  3) Counts + CSV export from the database (no video):
       python detect_plate.py --report

Bag version later: vehicle -> bag, vehicle type -> stitch OK/NG, plate -> bag number.
"""
import argparse
import csv
import json
import os
import queue
import re
import sqlite3
import threading
import time
from collections import Counter, deque
from datetime import datetime

import cv2
import numpy as np

START_COLOR = (0, 200, 0)      # BGR green
END_COLOR = (0, 0, 230)        # BGR red
STATE_COLOR = {"new": (160, 160, 160), "active": (0, 200, 255), "done": (0, 200, 0)}
PLATE_COLOR = (255, 120, 0)


# ----------------------------------------------------------------------------- helpers
def parse_source(src):
    """Return (cv2-ready source, is_live) for a webcam index, file path or stream URL."""
    s = int(src) if str(src).isdigit() else src
    return s, isinstance(s, int) or not os.path.isfile(str(s))


def open_source(src):
    cap = cv2.VideoCapture(parse_source(src)[0])
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {src}")
    return cap


def draw_text(img, text, org, scale=0.7, color=(255, 255, 255), thick=2):
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 3, y - h - 5), (x + w + 3, y + base), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _side(p, a, b):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def crossed(p0, p1, a, b):
    """True if the movement p0 -> p1 crosses the line segment a-b."""
    d1, d2 = _side(p0, a, b), _side(p1, a, b)
    d3, d4 = _side(a, p0, p1), _side(b, p0, p1)
    return d1 != 0 and d1 * d2 <= 0 and d3 * d4 <= 0


def between(p, start, end):
    """True if p is past START (on END's side) and before END (on START's side)."""
    mid = lambda l: ((l[0][0] + l[1][0]) / 2, (l[0][1] + l[1][1]) / 2)
    return (_side(p, *start) * _side(mid(end), *start) > 0
            and _side(p, *end) * _side(mid(start), *end) > 0)


def clamp_crop(frame, box, pad_x=0.0, pad_y=0.0):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = int(max(0, x1 - bw * pad_x)), int(min(w, x2 + bw * pad_x))
    y1, y2 = int(max(0, y1 - bh * pad_y)), int(min(h, y2 + bh * pad_y))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return frame[y1:y2, x1:x2].copy()


def plate_score(crop):
    """Higher = better crop for OCR: sharp and big."""
    h = crop.shape[0]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (max(1, int(g.shape[1] * 64 / h)), 64))
    sharp = cv2.Laplacian(g, cv2.CV_64F).var()
    return sharp * min(1.0, h / 40)


# ----------------------------------------------------------------------------- lines setup
def setup_lines(source, path):
    cap = open_source(source)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read a frame from {source}")
    h, w = frame.shape[:2]
    pts = []
    steps = ["Click START line: point 1", "Click START line: point 2",
             "Click END line: point 1", "Click END line: point 2",
             "S = save   R = redo   Q = quit"]

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))

    win = "Setup lines (vehicles go START -> END)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_click)
    while True:
        vis = frame.copy()
        draw_text(vis, steps[len(pts)], (20, 40), 0.9)
        for i, p in enumerate(pts):
            cv2.circle(vis, p, 7, START_COLOR if i < 2 else END_COLOR, -1)
        if len(pts) >= 2:
            cv2.line(vis, pts[0], pts[1], START_COLOR, 3)
            draw_text(vis, "START", (pts[0][0], pts[0][1] - 10), 0.7, START_COLOR)
        if len(pts) == 4:
            cv2.line(vis, pts[2], pts[3], END_COLOR, 3)
            draw_text(vis, "END", (pts[2][0], pts[2][1] - 10), 0.7, END_COLOR)
        cv2.imshow(win, vis)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("r"):
            pts.clear()
        elif k == ord("q"):
            print("Not saved.")
            break
        elif k == ord("s") and len(pts) == 4:
            data = {"start": [[x / w, y / h] for x, y in pts[:2]],
                    "end": [[x / w, y / h] for x, y in pts[2:]]}
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Saved lines to {path}")
            break
    cv2.destroyAllWindows()


def load_lines(path, w, h):
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Draw the lines first:\n"
                         f"  python detect_plate.py --source <video> --setup")
    d = json.load(open(path))
    to_px = lambda pts: [(int(x * w), int(y * h)) for x, y in pts]
    return to_px(d["start"]), to_px(d["end"])


# ----------------------------------------------------------------------------- plate text
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"}
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1", "J": "1",
            "Z": "2", "A": "4", "S": "5", "G": "6", "T": "7", "B": "8"}


# Indonesian region prefixes. Used as a soft preference, so a code missing here still reads.
REGION_CODES = set("""A B D E F G H K L M N P R S T W Z AA AB AD AE AG
BA BB BD BE BG BH BK BL BM BN BP DA DB DC DD DE DG DH DK DL DM DN DP DR DS DT DW
EA EB ED KB KH KT KU PA PB""".split())


def fix_plate(raw):
    """Fit OCR text to the Indonesian format: 1-2 letters, 1-4 digits, 0-3 letters.
    Fixes common confusions by position (O<->0, B<->8, ...). Returns 'B 1234 XYZ' or None."""
    s = re.sub(r"[^A-Z0-9]", "", raw.upper())
    best = None
    for p in (1, 2):
        for d in range(1, 5):
            if not 0 <= len(s) - p - d <= 3:
                continue
            out, changes = [], 0
            for i, ch in enumerate(s):
                if p <= i < p + d:
                    new = ch if ch.isdigit() else TO_DIGIT.get(ch)
                else:
                    new = ch if ch.isalpha() else TO_LETTER.get(ch)
                if new is None:
                    break
                changes += new != ch
                out.append(new)
            else:
                key = (changes + ("".join(out[:p]) not in REGION_CODES), -d)
                if best is None or key < best[0]:
                    t = "".join(out)
                    best = (key, f"{t[:p]} {t[p:p + d]} {t[p + d:]}".strip())
    return best[1] if best else None


def clean_text(raw):
    """Keep letters and digits only, as read. No format rules."""
    return re.sub(r"[^A-Z0-9]", "", raw.upper()) or None


def vote(readings, formatter=clean_text):
    """readings: [(plate_text, conf)] -> (text, agreement 0-1). Weighted per-character vote."""
    if not readings:
        return None, 0.0
    compact = [(t.replace(" ", ""), c) for t, c in readings]
    lens = Counter()
    for t, c in compact:
        lens[len(t)] += c
    n = lens.most_common(1)[0][0]
    same = [(t, c) for t, c in compact if len(t) == n]
    chars = []
    for i in range(n):
        cnt = Counter()
        for t, c in same:
            cnt[t[i]] += c
        chars.append(cnt.most_common(1)[0][0])
    final = formatter("".join(chars))
    if final is None:
        return None, 0.0
    agree = sum(t == final.replace(" ", "") for t, _ in compact) / len(compact)
    return final, agree


class PlateReader:
    ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    def __init__(self, gpu):
        import easyocr
        self.reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    def read(self, crop):
        h, w = crop.shape[:2]
        g = cv2.cvtColor(cv2.resize(crop, (int(w * 120 / h), 120), interpolation=cv2.INTER_CUBIC),
                         cv2.COLOR_BGR2GRAY)
        g = self.clahe.apply(g)
        res = self.reader.readtext(g, allowlist=self.ALLOW, detail=1, paragraph=False)
        if not res:
            return "", 0.0
        # Keep only the main row: the tallest text box and boxes on the same row.
        # This drops the small expiry-date row under Indonesian plate numbers.
        info = []
        for box, text, conf in res:
            ys = [p[1] for p in box]
            info.append((min(p[0] for p in box), min(ys), max(ys), text, conf))
        top = max(info, key=lambda r: r[2] - r[1])
        th = top[2] - top[1]
        main = [r for r in info
                if (r[2] - r[1]) >= 0.6 * th and top[1] <= (r[1] + r[2]) / 2 <= top[2]]
        main.sort(key=lambda r: r[0])
        return "".join(r[3] for r in main), float(np.mean([r[4] for r in main]))


# ----------------------------------------------------------------------------- database
SCHEMA = """CREATE TABLE IF NOT EXISTS vehicles (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at     TEXT,
    source        TEXT,
    track_id      INTEGER,
    vehicle_type  TEXT,
    plate         TEXT,
    plate_status  TEXT,     -- OK | REVIEW | NO_PLATE
    plate_conf    REAL,
    plate_agree   REAL,     -- share of OCR readings that agree with the final plate
    plate_reads   TEXT,     -- every reading, for checking
    completed     INTEGER,  -- 1 = crossed END line, 0 = lost after START line
    video_time_s  REAL,
    snapshot      TEXT,
    plate_image   TEXT
)"""


class Logger(threading.Thread):
    """Runs OCR + DB writes in the background so the video doesn't freeze."""

    def __init__(self, db_path, out_dir, reader, min_conf, formatter=clean_text):
        super().__init__(daemon=True)
        self.db_path, self.out_dir, self.reader, self.min_conf = db_path, out_dir, reader, min_conf
        self.formatter = formatter
        self.jobs = queue.Queue()
        self.recent = deque(maxlen=6)      # for the on-screen log
        self.plates = {}                   # track_id -> plate text, for labels
        self.rows = []                     # every saved row as a dict (used by the web app)

    def run(self):
        db = sqlite3.connect(self.db_path)
        db.execute(SCHEMA)
        db.commit()
        while True:
            job = self.jobs.get()
            if job is None:
                break
            try:
                self._process(db, job)
            except Exception as e:  # keep logging even if one record fails
                print(f"[logger] track {job['track_id']} failed: {e}")
        db.close()

    def _process(self, db, job):
        readings, raw = [], []
        for _, crop, _ in job["plates"]:
            text, conf = self.reader.read(crop) if self.reader else ("", 0.0)
            raw.append(f"{text}:{conf:.2f}")
            fixed = self.formatter(text) if text else None
            if fixed:
                readings.append((fixed, conf))
        plate, agree = vote(readings, self.formatter)
        conf = float(np.mean([c for _, c in readings])) if readings else 0.0
        if not job["plates"]:
            status = "NO_PLATE"
        elif plate and conf >= self.min_conf and (agree >= 0.5 or len(job["plates"]) == 1):
            status = "OK"
        else:
            status = "REVIEW"

        day = datetime.now().strftime("%Y%m%d")
        snap_dir = os.path.join(self.out_dir, "snapshots", day)
        os.makedirs(snap_dir, exist_ok=True)
        stamp = datetime.now().strftime("%H%M%S")
        base = os.path.join(snap_dir, f"{stamp}_id{job['track_id']}")
        snap = plate_img = None
        if job["snapshot"] is not None:
            snap = base + "_vehicle.jpg"
            cv2.imwrite(snap, job["snapshot"])
        if job["plates"]:
            plate_img = base + "_plate.jpg"
            cv2.imwrite(plate_img, job["plates"][0][1])

        row = dict(logged_at=datetime.now().isoformat(timespec="seconds"), source=job["source"],
                   track_id=job["track_id"], vehicle_type=job["vehicle_type"], plate=plate,
                   plate_status=status, plate_conf=round(conf, 3), plate_agree=round(agree, 3),
                   plate_reads=" | ".join(raw), completed=int(job["completed"]),
                   video_time_s=round(job["video_time"], 2), snapshot=snap, plate_image=plate_img)
        cur = db.execute(f"INSERT INTO vehicles ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                         tuple(row.values()))
        db.commit()
        self.rows.append(dict(id=cur.lastrowid, **row))

        label = plate or "?"
        self.plates[job["track_id"]] = label
        self.recent.appendleft(f"#{cur.lastrowid} {job['vehicle_type']} {label} [{status}]")
        print(f"[log] row {cur.lastrowid}: track {job['track_id']} | {job['vehicle_type']} | "
              f"{label} | {status} | conf {conf:.2f} agree {agree:.2f}"
              + ("" if job["completed"] else " | lost before END"))


def report(db_path, csv_path):
    if not os.path.exists(db_path):
        raise SystemExit(f"No database at {db_path} yet.")
    db = sqlite3.connect(db_path)
    print(f"\nDatabase: {db_path}")
    print("\nVehicles per type:")
    for t, n in db.execute("SELECT vehicle_type, COUNT(*) FROM vehicles GROUP BY 1 ORDER BY 2 DESC"):
        print(f"  {t:12s} {n}")
    print("\nPlate status:")
    for s, n in db.execute("SELECT plate_status, COUNT(*) FROM vehicles GROUP BY 1 ORDER BY 2 DESC"):
        print(f"  {s:12s} {n}")
    total = db.execute("SELECT COUNT(*) FROM vehicles").fetchone()[0]
    print(f"\nTotal: {total}")
    cur = db.execute("SELECT * FROM vehicles ORDER BY id")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([c[0] for c in cur.description])
        w.writerows(cur)
    print(f"Exported to {csv_path}")
    db.close()


# ----------------------------------------------------------------------------- tracking
class Track:
    def __init__(self, tid):
        self.id = tid
        self.state = "new"          # new -> active (crossed START) -> done (crossed END)
        self.types = Counter()
        self.prev_pt = None
        self.last_seen = 0.0
        self.plates = []            # [(score, plate_crop, vehicle_crop)] best first
        self.best_vehicle = (0, None)
        self.entered_inside = False # first seen between the lines (never crossed START)

    @property
    def vehicle_type(self):
        return self.types.most_common(1)[0][0] if self.types else "unknown"


class PlateCounter:
    """START -> END line logic for one video: tracks vehicles, keeps the sharpest plate crops
    and hands each finished vehicle to the Logger. Shared by the CLI and the web app (app.py)."""

    def __init__(self, model, start_line, end_line, logger, source, conf=0.25, imgsz=640,
                 tracker="bytetrack.yaml", device=None, top_k=5, lost_after=2.0):
        self.model, self.logger, self.source = model, logger, str(source)
        self.start_line, self.end_line = start_line, end_line
        self.conf, self.imgsz, self.tracker, self.device = conf, imgsz, tracker, device
        self.top_k, self.lost_after = top_k, lost_after       # lost_after: seconds unseen
        self.names = model.names
        self.plate_ids = {k for k, v in self.names.items() if "plate" in v.lower()}
        self.tracks, self.counts = {}, Counter()
        self.now = 0.0
        self.vehicles, self.plates, self.confs = [], [], []    # last frame's detections

    @property
    def active(self):
        """Vehicles currently between the lines."""
        return sum(tr.state == "active" for tr in self.tracks.values())

    def finalize(self, tr, completed):
        tr.state = "done"
        self.counts[tr.vehicle_type] += 1
        snap = tr.plates[0][2] if tr.plates else tr.best_vehicle[1]
        self.logger.jobs.put({
            "track_id": tr.id, "vehicle_type": tr.vehicle_type, "completed": completed,
            "plates": tr.plates, "snapshot": snap, "source": self.source, "video_time": self.now,
        })

    def update(self, frame, now):
        """Track one frame. `now` is seconds (video time for files, wall time for live)."""
        self.now = now
        res = self.model.track(frame, persist=True, tracker=self.tracker, conf=self.conf,
                               imgsz=self.imgsz, device=self.device, verbose=False)[0]
        b = res.boxes
        xyxy = b.xyxy.cpu().numpy() if len(b) else np.zeros((0, 4))
        cls = b.cls.int().cpu().tolist() if len(b) else []
        confs = b.conf.cpu().tolist() if len(b) else []
        ids = b.id.int().cpu().tolist() if b.id is not None else [None] * len(cls)

        vehicles, plates = [], []
        for box, c, tid in zip(xyxy, cls, ids):
            if c in self.plate_ids:
                plates.append(box)
            elif tid is not None:
                vehicles.append((tid, c, box))
        self.vehicles, self.plates, self.confs = vehicles, plates, confs
        tracks, start_line, end_line = self.tracks, self.start_line, self.end_line

        # --- update vehicle tracks and check line crossings
        for tid, c, box in vehicles:
            tr = tracks.setdefault(tid, Track(tid))
            tr.last_seen = now
            pt = ((box[0] + box[2]) / 2, box[3])          # bottom-center of the box
            if tr.state != "done":
                tr.types[self.names[c]] += 1
            if tr.prev_pt is None:
                # first seen already past START: the detector picked it up late
                if between(pt, start_line, end_line):
                    tr.state, tr.entered_inside = "active", True
            else:
                if tr.state == "new" and crossed(tr.prev_pt, pt, *start_line):
                    tr.state = "active"
                elif tr.state == "active" and crossed(tr.prev_pt, pt, *end_line):
                    self.finalize(tr, completed=True)
            tr.prev_pt = pt
            if tr.state == "active":
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area > tr.best_vehicle[0]:
                    tr.best_vehicle = (area, clamp_crop(frame, box))

        # --- give each plate to the smallest vehicle box that contains its center
        for pbox in plates:
            cx, cy = (pbox[0] + pbox[2]) / 2, (pbox[1] + pbox[3]) / 2
            owners = [(v[2][2] - v[2][0]) * (v[2][3] - v[2][1]) for v in vehicles]
            inside = [(a, v) for a, v in zip(owners, vehicles)
                      if v[2][0] <= cx <= v[2][2] and v[2][1] <= cy <= v[2][3]]
            if not inside:
                continue
            tid, _, vbox = min(inside, key=lambda x: x[0])[1]
            tr = tracks[tid]
            if tr.state != "active":
                continue
            crop = clamp_crop(frame, pbox, 0.08, 0.15)
            if crop is None or crop.shape[0] < 8:
                continue
            tr.plates.append((plate_score(crop), crop, clamp_crop(frame, vbox)))
            tr.plates.sort(key=lambda x: -x[0])
            del tr.plates[self.top_k:]

        # --- vehicles that disappeared
        for tid in list(tracks):
            tr = tracks[tid]
            if now - tr.last_seen > self.lost_after:
                # skip cars that started inside and never reached END (parked, false tracks)
                if tr.state == "active" and not tr.entered_inside:
                    self.finalize(tr, completed=False)
                del tracks[tid]

    def finish(self):
        """End of video: log vehicles still between the lines."""
        for tr in self.tracks.values():
            if tr.state == "active" and not tr.entered_inside:
                self.finalize(tr, completed=False)
        self.tracks.clear()

    def draw(self, frame, extra=()):
        """Frame with lines, boxes, counts and the recent log. `extra`: lines added on top."""
        H = frame.shape[0]
        vis = frame.copy()
        cv2.line(vis, *self.start_line, START_COLOR, 3)
        cv2.line(vis, *self.end_line, END_COLOR, 3)
        draw_text(vis, "START", (self.start_line[0][0], self.start_line[0][1] - 8), 0.6, START_COLOR)
        draw_text(vis, "END", (self.end_line[0][0], self.end_line[0][1] - 8), 0.6, END_COLOR)
        for pbox in self.plates:
            p = pbox.astype(int)
            cv2.rectangle(vis, (p[0], p[1]), (p[2], p[3]), PLATE_COLOR, 2)
        for tid, c, box in self.vehicles:
            tr = self.tracks.get(tid)
            if tr is None:
                continue
            x1, y1, x2, y2 = box.astype(int)
            col = STATE_COLOR[tr.state]
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            label = f"#{tid} {tr.vehicle_type}"
            if tr.state == "done":
                label += f" | {self.logger.plates.get(tid, 'reading...')}"
            draw_text(vis, label, (x1, max(20, y1 - 6)), 0.55, col)

        panel = list(extra) + [f"TOTAL {sum(self.counts.values())}"]
        panel += [f"{k}: {v}" for k, v in sorted(self.counts.items())]
        for i, line in enumerate(panel):
            draw_text(vis, line, (15, 30 + i * 28), 0.7)
        for i, line in enumerate(list(self.logger.recent)):
            draw_text(vis, line, (15, H - 20 - i * 26), 0.55, (200, 255, 200))
        return vis


def main():
    ap = argparse.ArgumentParser(description="Vehicle + license plate logger")
    ap.add_argument("--source", default="0", help="video file, webcam index or RTSP URL")
    ap.add_argument("--model", default="best.pt")
    ap.add_argument("--lines", default="lines.json")
    ap.add_argument("--setup", action="store_true", help="draw START and END lines, then exit")
    ap.add_argument("--report", action="store_true", help="print counts + export CSV, then exit")
    ap.add_argument("--db", default="vehicles.db")
    ap.add_argument("--out", default="output", help="folder for snapshots, CSV and video")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="e.g. 0 for GPU, cpu")
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--top-k", type=int, default=5, help="plate crops to OCR per vehicle")
    ap.add_argument("--min-ocr-conf", type=float, default=0.25)
    ap.add_argument("--no-ocr", action="store_true", help="skip OCR (test lines/counting only)")
    ap.add_argument("--id-format", action="store_true",
                    help="force Indonesian plate format (fix O/0, B/8 ...). Default: store text as read")
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "vehicles.csv")
    if args.setup:
        return setup_lines(args.source, args.lines)
    if args.report:
        return report(args.db, csv_path)

    from ultralytics import YOLO
    import torch

    model = YOLO(args.model)
    names = model.names
    plate_ids = {k for k, v in names.items() if "plate" in v.lower()}
    print("Plate class:", [names[k] for k in plate_ids] or "NONE (plates won't be read)")
    print("Vehicle classes:", [names[k] for k in set(names) - plate_ids])

    reader = None
    if not args.no_ocr and plate_ids:
        print("Loading OCR (first run downloads the EasyOCR model)...")
        reader = PlateReader(gpu=torch.cuda.is_available())
    logger = Logger(args.db, args.out, reader, args.min_ocr_conf,
                    fix_plate if args.id_format else clean_text)
    logger.start()

    cap = open_source(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    fps = fps if 1 < fps < 121 else 25
    is_file = not parse_source(args.source)[1]
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Could not read the first frame.")
    H, W = frame.shape[:2]
    start_line, end_line = load_lines(args.lines, W, H)
    counter = PlateCounter(model, start_line, end_line, logger, args.source, conf=args.conf,
                           imgsz=args.imgsz, tracker=args.tracker, device=args.device, top_k=args.top_k)

    writer = None
    if args.save_video:
        vid_path = os.path.join(args.out, f"annotated_{datetime.now():%Y%m%d_%H%M%S}.mp4")
        writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    frame_idx, t0, shown_fps = 0, time.time(), 0.0
    t_start = time.time()
    while ok:
        counter.update(frame, frame_idx / fps if is_file else time.time() - t_start)
        if writer is not None or not args.no_show:
            vis = counter.draw(frame, [f"FPS {shown_fps:.1f}"])
            if writer is not None:
                writer.write(vis)
            if not args.no_show:
                cv2.imshow("Plate logger  (Q = quit)", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        frame_idx += 1
        if frame_idx % 10 == 0:
            shown_fps = 10 / (time.time() - t0)
            t0 = time.time()
        ok, frame = cap.read()

    counter.finish()
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("Finishing OCR for the last vehicles...")
    logger.jobs.put(None)
    logger.join()

    print("\nThis run:")
    for k, v in sorted(counter.counts.items()):
        print(f"  {k:12s} {v}")
    print(f"  {'TOTAL':12s} {sum(counter.counts.values())}")
    report(args.db, csv_path)


if __name__ == "__main__":
    main()
