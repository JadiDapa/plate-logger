# Plate logger

Logs each vehicle once: it must cross the **START** line and then the **END** line.
For each vehicle it saves: row id, tracking id, vehicle type, plate text (EasyOCR),
plate status, time and snapshots. It also counts vehicles per type.

## Install (laptop)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
# For the CLI video window (detect_plate.py without --no-show), EasyOCR's "headless" OpenCV must go:
pip uninstall -y opencv-python-headless opencv-python
pip install opencv-python==5.0.0.93
```

NVIDIA GPU? Install the CUDA build of PyTorch first (https://pytorch.org/get-started/locally/).
Put your trained `best.pt` in this folder.

## Web UI (Streamlit)

```bash
streamlit run app.py
```

- **Sidebar**: video source (defaults to `vehicle-plate.mp4`; upload a file, or a path / RTSP URL / webcam index),
  model, confidence, OCR on/off, Indonesian plate format, plate crops per vehicle, min OCR confidence.
- **① Lines**: the lines start from `lines.json`. Click START (2 points), then END (2 points). Undo / Clear /
  Swap direction / Reset, and **Save** writes `lines.json` (used by the CLI too). You can also load or download it.
- **② Detect & dashboard**: live annotated preview while it runs, then KPIs, traffic over time, vehicles per type
  by plate status, OCR confidence, the vehicle log with plate crops (the plate column is editable to fix readings),
  a review queue, snapshots, the annotated video and downloads (CSV, ZIP with snapshots).

Each run is saved to `output/runs/<timestamp>/` (`vehicles.db`, `snapshots/`, `output.mp4`).
Changing a setting while a run is in progress stops it; vehicles logged so far are kept.

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repo. `.gitignore` keeps `best.pt` and `vehicle-plate.mp4`, and leaves out
   other videos, `output/` and `*.db`.
2. On https://share.streamlit.io: **Create app** → pick the repo/branch → main file `app.py`.
   In **Advanced settings**, choose Python **3.11 or 3.12** (`requirements.txt` is pinned for 3.11–3.14).
3. Deploy. The first build takes a few minutes (PyTorch CPU). The first OCR run downloads the EasyOCR model (~100 MB).

Notes for the cloud:
- It runs on a shared CPU, so expect a few fps. Turn OCR off to test lines and counting quickly.
- The disk is wiped on restart: download the CSV / ZIP after each run. Saving `lines.json` there is
  temporary too, so download the lines JSON (or commit it) to keep them.
- Webcams and cameras on your local network can't be reached from the cloud. Use uploads or public stream URLs.

To change dependencies, edit `requirements.in` and regenerate the pinned file:
`uv pip compile requirements.in --universal --python-version 3.11 -o requirements.txt`

## Run (CLI)

```bash
# 1. Draw the lines once per camera view: click START (2 points), END (2 points), press S
python detect_plate.py --source traffic.mp4 --setup

# 2. Run (Q to quit)
python detect_plate.py --source traffic.mp4 --model best.pt --save-video
python detect_plate.py --source 0                          # webcam
python detect_plate.py --source "rtsp://user:pass@ip/stream"

# 3. Counts + CSV from the database, any time
python detect_plate.py --report
```

Vehicles must move START → END. Reverse traffic is ignored.

## Output

| Where | What |
|---|---|
| `vehicles.db` | SQLite table `vehicles`, one row per vehicle |
| `output/vehicles.csv` | Same data, for Excel |
| `output/snapshots/<date>/` | Vehicle photo + best plate crop per row |
| `output/annotated_*.mp4` | Video with boxes, lines and counts (`--save-video`) |

`plate_status`:
- `OK`: the readings agree and confidence is decent
- `REVIEW`: unreadable or the readings disagree. Check the plate image.
- `NO_PLATE`: no plate was seen between the lines

`completed = 0` means the vehicle crossed START but was lost before END. It is still logged once.

## Useful options

| Option | Default | Use |
|---|---|---|
| `--conf` | 0.25 | Detection confidence |
| `--top-k` | 5 | Sharpest plate crops read per vehicle |
| `--min-ocr-conf` | 0.25 | Below this, the status becomes REVIEW |
| `--id-format` | off | Force the Indonesian plate format (fixes O/0, B/8). Off = store text as read |
| `--no-ocr` | off | Test lines and counting only |
| `--device` | auto | `0` for GPU, `cpu` |
| `--no-show` | off | No window (server / headless) |
