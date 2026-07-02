"""
BikePositionCV.py - DIY side-on bike fit analyzer (Windows / IMX415)
====================================================================

Markerless bike-fit tool in the spirit of MyVeloFit. Runs MediaPipe Pose on a
side-on view of you pedalling (live from your IMX415 4K USB cam, a standard
webcam, or a recorded clip) and tracks the five angles a road fit cares about.

It captures the frame at the BOTTOM OF THE PEDAL STROKE (where the knee is most
extended) and reports those angles against recommended road ranges, exactly
like the screenshot reference:

    - Knee Extension  (180 - knee bend)          -> SADDLE HEIGHT
    - Hip Open        (shoulder-hip-knee)        -> SADDLE FORE/AFT + REACH
    - Back Angle      (hip->shoulder vs horiz.)  -> REACH / BAR DROP
    - Shoulder Angle  (hip-shoulder-elbow)       -> REACH
    - Arm Angle       (shoulder-elbow-wrist)     -> REACH / COMFORT

A live side panel mirrors the reference UI: "Your measurements" colour-coded
green (in range) / orange (out of range), next to "Recommended Ranges" gauges.
On quit it prints a report with adjustment hints (tuned for a Canyon Ultimate
CF SL/SLX with SRAM AXS - all adjustments are stem/spacer/saddle, no cabling
to worry about).

SETUP
-----
    pip install opencv-python "mediapipe==0.10.20" numpy
    # NOTE: pin mediapipe==0.10.20. The newest PyPI release (0.10.35) ships a
    # broken Windows wheel with no native bindings -> "module 'mediapipe' has
    # no attribute 'solutions'". 0.10.20 is the last good cp3x Windows build.

    # Optional hands-free voice control (say "start"/"stop"):
    pip install vosk piper-tts pyttsx3 sounddevice
    # Both offline models (Vosk recognizer + Piper natural voice) auto-download
    # on the first --voice run. pyttsx3 is just a fallback if Piper is missing.

CAMERA RIG (this matters more than the model)
    - Tripod, lens height ~ bottom-bracket / crank-axle height.
    - Camera axis exactly perpendicular to the bike's plane (dead side-on).
    - Stand BACK 3-4 m and let the 4K sensor frame it. Don't get close with a
      wide lens -> barrel distortion bends limbs and corrupts angles.
    - Even lighting; rider in contrasting clothing helps the model.
    - Pedal steadily for ~20-30 s so a clean bottom-of-stroke frame is caught.

USAGE
    # live on the IMX415 (auto-detected), drive (right) side toward camera:
    python BikePositionCV.py --side right

    # force a specific camera index / a normal webcam:
    python BikePositionCV.py --source 0 --side right

    # analyse a recorded clip:
    python BikePositionCV.py --source ride.mp4 --side left

    # label a run up front (skips the end-of-run prompt):
    python BikePositionCV.py --side right --name "5mm saddle lowered"

    # open the analysis dashboard of all saved runs (browser):
    python BikePositionCV.py --analyze

    # hands-free (no reaching for the keyboard mid-test):
    python BikePositionCV.py --side right --voice
    #   say "start" -> it replies "recording started" and begins the test
    #   say "stop"  -> it replies "recording stopped" and freezes the capture
    python BikePositionCV.py --list-mics          # find a mic index if needed
    python BikePositionCV.py --voice --mic 2      # use a specific mic

    Keys:  SPACE = start/pause recording (capture only happens while recording,
                   so mounting/dismounting can't poison the reading)
           R     = reset captured bottom-of-stroke
           Q     = quit + print report

MEASUREMENT
    Every pedal stroke is sampled at its bottom (ankle lowest), so each angle is
    reported as mean / median / std / CV / min-max over N strokes - not a single
    noisy frame. The headline value is the MEDIAN (robust to outliers).

SAVED FILES (each run gets its own folder ./captures/<timestamp>_<name>/)
    clip<N>.mp4      annotated video, one per record session
    bottom.jpg       a representative bottom-of-stroke frame
    report.txt       fit report with per-angle stats + adjustment hints
    report.csv       median/mean/std/CV/min/max/n per angle, for spreadsheets
    run.json         structured record (angles, stats, targets, name, note)

ANALYSIS
    python BikePositionCV.py --analyze
    Builds captures/analysis.html (and opens it): a table of every run with
    angles colour-coded vs target + deltas to the previous run, plus trend
    charts per angle. Legacy folders that only have a report.csv are included
    automatically.
"""

import argparse
import base64
import csv
import io
import json
import os
import queue
import threading
import urllib.parse
import webbrowser
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# MediaPipe Pose landmark indices, per body side.
L = dict(shoulder=11, elbow=13, wrist=15, hip=23, knee=25, ankle=27)
R = dict(shoulder=12, elbow=14, wrist=16, hip=24, knee=26, ankle=28)

# Reference windows for the bottom-of-stroke key frame (road / endurance).
# (key, label, lo, hi, what-it-tunes). Treat as starting points, not gospel.
METRICS = [
    ("knee_ext", "Knee Extension", 32,  38,  "Saddle height"),
    ("hip_open", "Hip Open",       96,  104, "Saddle fore/aft + reach"),
    ("back",     "Back Angle",     37,  43,  "Reach / bar drop"),
    ("shoulder", "Shoulder Angle", 86,  93,  "Reach"),
    ("arm",      "Arm Angle",      145, 175, "Reach / arm comfort"),
]

# BGR colours
GREEN  = (0, 200, 0)
ORANGE = (0, 165, 255)
RED    = (0, 0, 255)
CYAN   = (255, 200, 0)
WHITE  = (255, 255, 255)
GREY   = (150, 150, 150)
PANEL  = (30, 22, 16)   # dark navy-ish background like the reference


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def angle_3pt(a, b, c):
    """Interior angle at b (degrees), formed by points a-b-c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    v1, v2 = a - b, c - b
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))


def angle_to_horizontal(a, b):
    """Acute angle (degrees) of the line a->b relative to horizontal."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    return float(abs(np.degrees(np.arctan2(abs(dy), abs(dx)))))


def px(landmarks, idx, w, h):
    lm = landmarks[idx]
    return (lm.x * w, lm.y * h)


def compute_angles(p):
    """Given a point-getter p(name)->(x,y), return the five fit angles."""
    hip, knee, ankle = p("hip"), p("knee"), p("ankle")
    sh, elb, wr = p("shoulder"), p("elbow"), p("wrist")
    knee_interior = angle_3pt(hip, knee, ankle)
    return {
        "knee_ext": 180.0 - knee_interior,           # 0 = dead straight leg
        "hip_open": angle_3pt(sh, hip, knee),
        "back":     angle_to_horizontal(hip, sh),
        "shoulder": angle_3pt(hip, sh, elb),
        "arm":      angle_3pt(sh, elb, wr),
        "_knee_interior": knee_interior,
    }


def metric_stats(values):
    """Descriptive stats over a list of per-stroke samples (degrees)."""
    a = np.array([v for v in values if v is not None], dtype=float)
    if a.size == 0:
        return None
    mean = float(np.mean(a))
    std = float(np.std(a, ddof=1)) if a.size > 1 else 0.0
    return {
        "n":      int(a.size),
        "mean":   round(mean, 1),
        "median": round(float(np.median(a)), 1),
        "std":    round(std, 2),
        "min":    round(float(a.min()), 1),
        "max":    round(float(a.max()), 1),
        "cv":     round(100.0 * std / mean, 1) if mean else 0.0,  # % variability
    }


# --------------------------------------------------------------------------- #
# Camera (IMX415 on Windows, with graceful fallback)
# --------------------------------------------------------------------------- #
def find_imx415():
    """Return the index of the 4K IMX415, or None if not found."""
    for i in [0, 1, 2, 3]:
        cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FOURCC,      cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  3840)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        if w == 3840:
            print(f"IMX415 found at index {i}")
            return i
    return None


def open_camera(source, width, height):
    """Open a camera index (IMX415-tuned) or a video file path."""
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_MSMF)
        cap.set(cv2.CAP_PROP_FOURCC,      cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS,          60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        cap.read()   # prime once - fixes a Windows MSMF timing quirk
    else:
        cap = cv2.VideoCapture(source)
    return cap


def open_video_writer(path, fps, size):
    """Return a VideoWriter that produces BROWSER-PLAYABLE H.264 .mp4 via the
    Windows Media Foundation (MSMF) backend. OpenCV's bundled FFmpeg can't encode
    H.264 (no openh264/libx264), and its 'mp4v' codec won't play in browsers, so
    MSMF+H264 is the reliable path. Falls back to mp4v only if MF is unavailable.
    Note: H.264 needs even width/height."""
    try:
        w = cv2.VideoWriter(path, cv2.CAP_MSMF,
                            cv2.VideoWriter_fourcc(*"H264"), fps, size)
        if w.isOpened():
            return w
        w.release()
    except Exception:
        pass
    print("  (H.264/MSMF unavailable - clip saved as mp4v, may not play in browser)")
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)


# --------------------------------------------------------------------------- #
# Rendering - measurements panel that mirrors the reference UI
# --------------------------------------------------------------------------- #
def gauge_color(val, lo, hi):
    if val is None:
        return GREY
    span = hi - lo
    if lo <= val <= hi:
        return GREEN
    # within half a window of the edge -> orange, beyond -> red
    if val < lo - 0.5 * span or val > hi + 0.5 * span:
        return RED
    return ORANGE


def draw_gauge(img, x, y, w, h, lo, hi, val):
    """Red-yellow-green gradient bar with the recommended window in the centre
    and a tick at the measured value."""
    # The drawn axis spans [lo - span, hi + span]; recommended window sits mid.
    span = hi - lo
    axis_lo, axis_hi = lo - span, hi + span
    for i in range(w):
        t = i / max(w - 1, 1)            # 0..1 across the bar
        # distance from the centre of the recommended window, normalised
        center = 0.5
        d = abs(t - center) / 0.5        # 0 at centre, 1 at edges
        if d < 0.34:
            col = GREEN
        elif d < 0.67:
            col = ORANGE
        else:
            col = RED
        cv2.line(img, (x + i, y), (x + i, y + h), col, 1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), 1)
    # value tick
    if val is not None:
        t = (val - axis_lo) / max(axis_hi - axis_lo, 1e-6)
        t = min(max(t, 0.0), 1.0)
        tx = int(x + t * w)
        cv2.line(img, (tx, y - 3), (tx, y + h + 3), WHITE, 2)


def render_panel(panel_w, panel_h, captured, live, recording, n_strokes=0):
    """Build the side panel image (measurements + recommended ranges).
    When strokes have been sampled, the values shown are the running MEDIAN
    across strokes (robust to single bad frames)."""
    panel = np.full((panel_h, panel_w, 3), PANEL, np.uint8)
    cv2.putText(panel, "Bottom of Pedal Stroke", (24, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)

    src = captured if captured else live
    if recording:
        cv2.circle(panel, (32, 61), 7, RED, -1)
        cv2.putText(panel, f"REC - median of {n_strokes} strokes", (48, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)
    else:
        state = (f"CAPTURED - median of {n_strokes} strokes" if captured
                 else "PAUSED - press SPACE to record")
        cv2.putText(panel, state, (24, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY, 1)

    cv2.putText(panel, "Your measurements", (24, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    cv2.putText(panel, "Recommended Ranges", (290, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

    y = 150
    for key, label, lo, hi, _ in METRICS:
        val = src.get(key) if src else None
        col = gauge_color(val, lo, hi)
        cv2.putText(panel, label, (24, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        vtxt = f"{val:.0f}deg" if val is not None else "--"
        cv2.putText(panel, vtxt, (200, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        # ranges + gauge
        cv2.putText(panel, f"{lo:.0f}", (290, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        draw_gauge(panel, 326, y - 14, 120, 16, lo, hi, val)
        cv2.putText(panel, f"{hi:.0f}", (452, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        y += 46

    cv2.putText(panel, "SPACE rec/pause  R reset  Q quit+report",
                (24, panel_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY, 1)
    return panel


def draw_skeleton(frame, lm_proto, side, angles):
    mp_draw.draw_landmarks(frame, lm_proto, mp_pose.POSE_CONNECTIONS)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
# NOTE on knee_ext direction: knee_ext = 180 - interior knee angle, i.e. how
# much the knee is BENT at the bottom of the stroke. More bend (HIGH) = saddle
# too low; less bend / straighter leg (LOW) = saddle too high.
LOW_HINTS = {
    "knee_ext": "leg too straight / over-extended -> LOWER saddle",
    "hip_open": "hip too closed -> shift saddle back / raise bars (open hip)",
    "back":     "back quite flat/low -> raise front (shorter stem / +spacers)",
    "shoulder": "shoulder closed / not reaching enough -> lengthen or lower reach",
    "arm":      "arms too bent -> lengthen reach (longer stem / lower bars)",
}
HIGH_HINTS = {
    "knee_ext": "knee too bent at bottom / hips may rock -> RAISE saddle",
    "hip_open": "hip very open / upright -> can drop bars or lengthen reach",
    "back":     "back quite upright -> can drop bars (-spacers / lower stem)",
    "shoulder": "shoulder very open / reaching too far -> shorten reach",
    "arm":      "arms too straight/locked -> shorten reach or bend elbows",
}


def build_report(angles, stats=None):
    """Return the report as a list of text lines. `angles` holds the median per
    metric; `stats` (optional) holds {metric: {n,mean,median,std,min,max,cv}}."""
    lines = ["=" * 66,
             " BIKE FIT REPORT - Canyon Ultimate CF (bottom of pedal stroke)",
             "=" * 66]
    if not angles:
        lines += ["  No strokes captured. Pedal a few full strokes in side view.",
                  "=" * 66]
        return lines
    n = max((((stats or {}).get(k) or {}).get("n", 0)) for k, *_ in METRICS) \
        if stats else 0
    if n:
        lines.append(f"  Aggregated over {n} pedal strokes "
                     f"(value shown = median).")
        lines.append("-" * 66)
    for key, label, lo, hi, what in METRICS:
        val = angles.get(key)
        if val is None:
            lines.append(f"  {label:15s}: no data")
            continue
        if val < lo:
            verdict, hint = "LOW ", LOW_HINTS[key]
        elif val > hi:
            verdict, hint = "HIGH", HIGH_HINTS[key]
        else:
            verdict, hint = "OK  ", ""
        lines.append(f"  {label:15s}: {val:6.1f}deg   target {lo}-{hi}   "
                     f"[{verdict}] -> {what}")
        st = (stats or {}).get(key)
        if st:
            lines.append(f"      mean {st['mean']:.1f}  median {st['median']:.1f}  "
                         f"sd {st['std']:.2f}  CV {st['cv']:.1f}%  "
                         f"n {st['n']}  range {st['min']:.1f}-{st['max']:.1f}")
        if hint:
            lines.append(f"      | {hint}")
    lines += ["=" * 66,
              "  Saddle TILT can't be read side-on - set near level, tune by feel.",
              "  Change ONE thing at a time, then re-measure.",
              "  SD/CV show how steady your pedalling was - lower is more reliable."]
    return lines


def fit_report(angles, stats=None, txt_path=None, csv_path=None):
    lines = build_report(angles, stats)
    print("\n" + "\n".join(lines) + "\n")

    if txt_path:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  Report saved: {txt_path}")

    if csv_path and angles:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["measurement", "median_deg", "mean_deg", "std_deg",
                        "cv_pct", "min_deg", "max_deg", "n_strokes",
                        "target_lo", "target_hi", "verdict"])
            for key, label, lo, hi, _ in METRICS:
                val = angles.get(key)
                if val is None:
                    w.writerow([label, "", "", "", "", "", "", "", lo, hi, "no_data"])
                    continue
                v = "OK" if lo <= val <= hi else ("LOW" if val < lo else "HIGH")
                st = (stats or {}).get(key) or {}
                w.writerow([label, f"{val:.1f}", st.get("mean", ""),
                            st.get("std", ""), st.get("cv", ""),
                            st.get("min", ""), st.get("max", ""),
                            st.get("n", ""), lo, hi, v])
        print(f"  CSV saved:    {csv_path}")


# --------------------------------------------------------------------------- #
# Voice control (offline: Vosk speech-to-text + Piper/SAPI text-to-speech)
# --------------------------------------------------------------------------- #
VOSK_MODEL_NAME = "vosk-model-small-en-us-0.15"
VOSK_MODEL_URL = ("https://alphacephei.com/vosk/models/"
                  "vosk-model-small-en-us-0.15.zip")
PIPER_VOICE_NAME = "en_US-amy-medium"   # natural offline neural voice
START_WORDS = ("start", "begin", "record", "go")
STOP_WORDS = ("stop", "end", "finish", "done")


class Speaker:
    """Non-blocking text-to-speech on a background thread (so it never stalls
    the video loop). Prefers Piper (natural, offline neural voice); falls back
    to Windows SAPI (pyttsx3), then to silent text-only."""

    def __init__(self, enabled=True, piper_model=None):
        self.enabled = enabled
        self.q = queue.Queue()
        self.backend = None
        if not enabled:
            return

        # 1) Piper - natural neural voice, fully offline.
        if piper_model and os.path.isfile(piper_model):
            try:
                from piper import PiperVoice
                import sounddevice  # noqa: F401  (ensure playback is available)
                self._voice = PiperVoice.load(piper_model)
                self.backend = "piper"
            except Exception as e:                   # pragma: no cover
                print(f"[voice] Piper unavailable ({e}); trying SAPI voice.")

        # 2) Windows SAPI fallback.
        if self.backend is None:
            try:
                import pyttsx3
                self._pyttsx3 = pyttsx3
                self.backend = "pyttsx3"
            except Exception as e:                   # pragma: no cover
                print(f"[voice] TTS unavailable ({e}); spoken replies disabled.")
                self.enabled = False
                return

        threading.Thread(target=self._run, daemon=True).start()
        print(f"[voice] TTS voice: "
              f"{'Piper (' + PIPER_VOICE_NAME + ')' if self.backend == 'piper' else 'Windows SAPI'}")

    def _run(self):
        if self.backend == "piper":
            import sounddevice as sd
            sr = self._voice.config.sample_rate
            while True:
                text = self.q.get()
                if text is None:
                    break
                try:
                    parts = [c.audio_int16_array
                             for c in self._voice.synthesize(text)]
                    if parts:
                        sd.play(np.concatenate(parts), sr)
                        sd.wait()
                except Exception:                    # pragma: no cover
                    pass
        else:
            engine = self._pyttsx3.init()
            engine.setProperty("rate", 180)
            while True:
                text = self.q.get()
                if text is None:
                    break
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception:                    # pragma: no cover
                    pass

    def say(self, text):
        if self.enabled:
            self.q.put(text)


def resolve_piper_model(path):
    """Return a usable Piper .onnx voice, downloading it once if missing."""
    if os.path.isfile(path):
        return path
    import subprocess
    import sys
    pdir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(pdir, exist_ok=True)
    print(f"[voice] Piper voice not found; downloading {PIPER_VOICE_NAME} "
          f"(~60 MB, one time)...")
    try:
        subprocess.run([sys.executable, "-m", "piper.download_voices",
                        PIPER_VOICE_NAME, "--download-dir", pdir], check=True)
    except Exception as e:                           # pragma: no cover
        print(f"[voice] Piper voice download failed ({e}); will use SAPI voice.")
        return None
    return path if os.path.isfile(path) else None


def resolve_vosk_model(path):
    """Return a usable model dir, downloading the small EN model if missing."""
    if os.path.isdir(path):
        return path
    import urllib.request
    import zipfile
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)
    print(f"[voice] Vosk model not found at {path}")
    print(f"[voice] Downloading {VOSK_MODEL_URL} (~40 MB, one time)...")
    zpath = os.path.join(parent, "_vosk_model.zip")
    try:
        urllib.request.urlretrieve(VOSK_MODEL_URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(parent)
        os.remove(zpath)
    except Exception as e:                           # pragma: no cover
        print(f"[voice] Model download failed: {e}")
        return None
    extracted = os.path.join(parent, VOSK_MODEL_NAME)
    return extracted if os.path.isdir(extracted) else None


def start_voice_listener(cmd_queue, model_path, device=None):
    """Spawn a daemon thread that listens on the mic and pushes 'start'/'stop'
    onto cmd_queue. Returns True if it started, False otherwise."""
    try:
        import json
        import sounddevice as sd
        from vosk import Model, KaldiRecognizer, SetLogLevel
    except Exception as e:                           # pragma: no cover
        print(f"[voice] speech recognition unavailable ({e}).")
        return False

    SetLogLevel(-1)  # silence vosk's verbose stderr logging
    model = Model(model_path)
    vocab = json.dumps(list(START_WORDS) + list(STOP_WORDS) + ["[unk]"])
    rec = KaldiRecognizer(model, 16000, vocab)

    def worker():
        def callback(indata, frames, time_, status):
            if not rec.AcceptWaveform(bytes(indata)):
                return
            words = set(json.loads(rec.Result()).get("text", "").split())
            if words & set(STOP_WORDS):
                cmd_queue.put("stop")
            elif words & set(START_WORDS):
                cmd_queue.put("start")

        with sd.RawInputStream(samplerate=16000, blocksize=8000, device=device,
                               dtype="int16", channels=1, callback=callback):
            while True:
                sd.sleep(200)

    threading.Thread(target=worker, daemon=True).start()
    return True


# --------------------------------------------------------------------------- #
# Storage + analysis (per-run folders, run.json, HTML dashboard)
# --------------------------------------------------------------------------- #
BIKE_NAME = "Canyon Ultimate CF"


def slugify(text):
    s = "".join(c if (c.isalnum() or c in " -_") else "" for c in (text or ""))
    s = "_".join(s.split())
    return s[:40] or "run"


def write_run_json(run_dir, session, name, note, side, angles, stats=None):
    rec = {
        "session": session,
        "datetime": datetime.now().isoformat(timespec="seconds"),
        "name": name or session,
        "note": note or "",
        "side": side,
        "bike": BIKE_NAME,
        "n_strokes": max((((stats or {}).get(k) or {}).get("n", 0))
                         for k, *_ in METRICS) if stats else 0,
        "angles": {k: (round(angles[k], 1) if angles and angles.get(k) is not None
                       else None)
                   for k, *_ in METRICS},
        "stats": {k: (stats or {}).get(k) for k, *_ in METRICS} if stats else {},
        "targets": {k: [lo, hi] for k, _, lo, hi, _ in METRICS},
    }
    with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
    return rec


def _run_from_csv(d, entry):
    """Build a run record from a legacy folder that only has a *report.csv."""
    csvs = [f for f in os.listdir(d) if f.lower().endswith(".csv")]
    if not csvs:
        return None
    label2key = {label: key for key, label, *_ in METRICS}
    angles = {}
    with open(os.path.join(d, csvs[0]), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = label2key.get(row.get("measurement", ""))
            val = row.get("median_deg") or row.get("value_deg", "")
            if key and val:
                try:
                    angles[key] = float(val)
                except ValueError:
                    pass
    if not angles:
        return None
    mtime = datetime.fromtimestamp(os.path.getmtime(d))
    return {
        "name": entry,
        "note": "",
        "datetime": mtime.isoformat(timespec="seconds"),
        "angles": angles,
        "dir": d,
    }


def load_runs(outdir):
    """Return all runs (newest last), reading run.json or falling back to CSV."""
    runs = []
    if not os.path.isdir(outdir):
        return runs
    for entry in sorted(os.listdir(outdir)):
        d = os.path.join(outdir, entry)
        if not os.path.isdir(d):
            continue
        jpath = os.path.join(d, "run.json")
        rec = None
        if os.path.isfile(jpath):
            try:
                with open(jpath, encoding="utf-8") as f:
                    rec = json.load(f)
                rec["dir"] = d
            except Exception:
                rec = None
        if rec is None:
            rec = _run_from_csv(d, entry)
        if rec:
            runs.append(rec)
    runs.sort(key=lambda r: r.get("datetime") or "")
    return runs


def _band_dist(val, lo, hi):
    """Distance from the target band (0 if inside); None if no value."""
    if val is None:
        return None
    if val < lo:
        return lo - val
    if val > hi:
        return val - hi
    return 0.0


def _verdict(val, lo, hi):
    if val is None:
        return "nodata"
    if lo <= val <= hi:
        return "ok"
    span = hi - lo
    if val < lo - 0.5 * span or val > hi + 0.5 * span:
        return "far"
    return "near"


def make_trend_png(runs):
    """Render per-metric trend charts across runs to a base64 PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    labels = [r.get("name") or (r.get("datetime") or "")[:16] for r in runs]
    panel = "#12161c"
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), facecolor=panel)
    axes = axes.flatten()
    for ax, (key, label, lo, hi, _) in zip(axes, METRICS):
        ax.set_facecolor(panel)
        ys = [(r.get("angles") or {}).get(key) for r in runs]
        es = [(((r.get("stats") or {}).get(key) or {}).get("std")) for r in runs]
        ax.axhspan(lo, hi, color="#37c837", alpha=0.18)
        xs = [i for i, y in enumerate(ys) if y is not None]
        yv = [y for y in ys if y is not None]
        ye = [es[i] if es[i] is not None else 0.0 for i in xs]
        ax.plot(xs, yv, "-", color="#5b9cff", zorder=1, linewidth=1.6)
        if any(e > 0 for e in ye):
            ax.errorbar(xs, yv, yerr=ye, fmt="none", ecolor="#7f93a8",
                        elinewidth=1, capsize=3, zorder=2)
        for i, y in zip(xs, yv):
            v = _verdict(y, lo, hi)
            c = {"ok": "#37c837", "near": "#ff8c00", "far": "#e53935"}.get(v, "#888")
            ax.scatter([i], [y], color=c, s=45, zorder=3,
                       edgecolors=panel, linewidths=1)
        ax.set_title(f"{label}  (target {lo}-{hi})", fontsize=10, color="#dce6f0")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7,
                           color="#9fb0c2")
        ax.tick_params(colors="#9fb0c2")
        for sp in ax.spines.values():
            sp.set_color("#2a2f35")
        ax.grid(alpha=0.12, color="#9fb0c2")
    axes[-1].axis("off")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=panel)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


_CELL_COLORS = {"ok": "#1e6b2e", "near": "#8a5a00", "far": "#7a1f1f",
                "nodata": "#333"}


def _run_file_links(run_dir):
    """Build <a> links to a run's artifacts (video / screenshot / csv / txt),
    relative to analysis.html (which sits one level up, in the captures root)."""
    if not run_dir or not os.path.isdir(run_dir):
        return "<span class=muted>&mdash;</span>"
    try:
        entries = sorted(os.listdir(run_dir))
    except OSError:
        return "<span class=muted>&mdash;</span>"
    base = os.path.basename(run_dir.rstrip("/\\"))

    def href(fname):
        return urllib.parse.quote(base) + "/" + urllib.parse.quote(fname)

    def pick(exts, prefer=None):
        cands = [f for f in entries if f.lower().endswith(exts)]
        if prefer:
            for f in cands:
                if prefer in f.lower():
                    return [f]
        return cands

    links = []
    vids = pick((".mp4", ".avi", ".mov"))
    for i, v in enumerate(vids):
        tag = f"video{i + 1}" if len(vids) > 1 else "video"
        links.append(f"<a class=fl href='{href(v)}' target=_blank>&#9658; {tag}</a>")
    shot = pick((".jpg", ".jpeg", ".png"), prefer="bottom") or pick((".jpg", ".jpeg", ".png"))
    if shot:
        links.append(f"<a class=fl href='{href(shot[0])}' target=_blank>&#128247; image</a>")
    csvf = pick((".csv",))
    if csvf:
        links.append(f"<a class=fl href='{href(csvf[0])}' target=_blank>&#8623; csv</a>")
    txtf = pick((".txt",), prefer="report") or pick((".txt",))
    if txtf:
        links.append(f"<a class=fl href='{href(txtf[0])}' target=_blank>&#9776; text</a>")
    return "".join(links) or "<span class=muted>&mdash;</span>"


def build_dashboard_html(runs, png_b64, bg_uri=None):
    cols = "".join(f"<th>{label}<br><span class=t>{lo}-{hi}</span></th>"
                   for _, label, lo, hi, _ in METRICS)
    rows = []
    prev = {}
    for r in runs:
        ang = r.get("angles") or {}
        st_all = r.get("stats") or {}
        cells = []
        for key, _, lo, hi, _ in METRICS:
            val = ang.get(key)
            v = _verdict(val, lo, hi)
            txt = f"{val:.0f}&deg;" if val is not None else "&mdash;"
            st = st_all.get(key) or {}
            sd = ""
            tip = ""
            if st:
                sd = f"<div class=sd>&plusmn;{st.get('std', 0):.1f}</div>"
                tip = (f"mean {st.get('mean')}  median {st.get('median')}  "
                       f"sd {st.get('std')}  CV {st.get('cv')}%  "
                       f"n {st.get('n')}  range {st.get('min')}-{st.get('max')}")
            delta = ""
            if val is not None and prev.get(key) is not None:
                d = val - prev[key]
                dn = _band_dist(val, lo, hi)
                dp = _band_dist(prev[key], lo, hi)
                good = dn is not None and dp is not None and dn < dp - 0.05
                worse = dn is not None and dp is not None and dn > dp + 0.05
                arrow = "&#9650;" if d > 0 else ("&#9660;" if d < 0 else "")
                dcol = "#5fd35f" if good else ("#ff6b6b" if worse else "#aaa")
                delta = (f"<div class=d style='color:{dcol}'>{arrow}"
                         f"{abs(d):.1f}</div>")
            cells.append(f"<td title='{tip}' style='background:{_CELL_COLORS[v]}'>"
                         f"{txt}{sd}{delta}</td>")
        prev = {k: ang.get(k) for k, *_ in METRICS}
        date = (r.get("datetime") or "")[:16].replace("T", " ")
        note = (r.get("note") or "").replace("<", "&lt;")
        n = r.get("n_strokes") or (st_all.get("knee_ext") or {}).get("n") or ""
        files = _run_file_links(r.get("dir"))
        rows.append(f"<tr><td class=dt>{date}</td><td class=nm>{r.get('name','')}"
                    f"</td><td class=n>{n}</td>{''.join(cells)}"
                    f"<td class=note>{note}</td><td class=files>{files}</td></tr>")
    img = (f"<img src='data:image/png;base64,{png_b64}'/>" if png_b64
           else "<p class=muted>(matplotlib not available - charts skipped)</p>")
    bg = (f"linear-gradient(rgba(8,10,13,.86),rgba(8,10,13,.95)),url('{bg_uri}')"
          if bg_uri else "#0d1014")
    latest = (runs[-1].get("name") or "") if runs else "-"
    return f"""<!doctype html><html><head><meta charset=utf-8>
<title>Bike Fit Analysis</title><style>
*{{box-sizing:border-box}}
body{{margin:0;min-height:100vh;color:#e8edf4;
 font-family:'Segoe UI',-apple-system,Arial,sans-serif;
 background:{bg};background-size:cover;background-position:center;
 background-attachment:fixed}}
.wrap{{max-width:1080px;margin:0 auto;padding:40px 26px 60px}}
.hero{{padding:6px 0 18px;border-bottom:1px solid rgba(255,255,255,.08);
 margin-bottom:6px}}
.hero h1{{margin:0;font-size:30px;font-weight:600;letter-spacing:.3px}}
.hero .sub{{margin:8px 0 0;color:#aeb9c6;font-size:14px;max-width:780px;
 line-height:1.55}}
.meta{{display:flex;gap:10px;margin:16px 0 0;flex-wrap:wrap}}
.chip{{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08);
 border-radius:999px;padding:5px 13px;font-size:12px;color:#cdd9e6}}
.chip b{{color:#fff;font-weight:600}}
.card{{background:rgba(20,24,30,.72);border:1px solid rgba(255,255,255,.07);
 border-radius:14px;padding:18px 18px 20px;margin-top:22px}}
.card h2{{margin:2px 0 4px;font-size:16px;font-weight:600;color:#dce6f0}}
.card .note{{color:#8fa0b2;font-size:12.5px;margin:0 0 4px}}
table{{border-collapse:collapse;width:100%;font-size:13.5px;margin-top:10px}}
th,td{{padding:9px 10px;text-align:center}}
th{{color:#9fb0c2;font-weight:600;font-size:12px;
 border-bottom:1px solid rgba(255,255,255,.12)}}
th .t{{color:#6f8093;font-weight:400;font-size:10.5px}}
tbody tr{{border-bottom:1px solid rgba(255,255,255,.05)}}
tbody tr:hover{{background:rgba(255,255,255,.03)}}
td.dt{{color:#9fb0c2;white-space:nowrap;text-align:left}}
td.nm{{text-align:left;font-weight:600}} td.n{{color:#9fb0c2}}
td.note{{text-align:left;color:#c4d0dd;max-width:240px}}
td.files{{text-align:left;white-space:nowrap}}
a.fl{{display:inline-block;margin:2px 4px 2px 0;padding:3px 9px;font-size:11.5px;
 color:#cdd9e6;text-decoration:none;background:rgba(255,255,255,.05);
 border:1px solid rgba(255,255,255,.10);border-radius:7px}}
a.fl:hover{{background:rgba(106,163,255,.18);border-color:rgba(106,163,255,.5);
 color:#fff}}
td .d{{font-size:11px;margin-top:2px}}
td .sd{{font-size:10px;color:#9fb0c2;margin-top:1px}}
td[title]{{cursor:help}}
img{{max-width:100%;border-radius:10px;margin-top:6px}}
.muted{{color:#8fa0b2}}
.legend{{margin-top:12px}}
.legend span{{display:inline-block;padding:3px 11px;border-radius:6px;
 margin-right:8px;font-size:12px}}
</style></head><body><div class=wrap>
<div class=hero>
 <h1>Bike Fit Analysis</h1>
 <p class=sub>{BIKE_NAME} &middot; side-on, bottom of pedal stroke. Each value is
 the <b>median across pedal strokes</b> (&plusmn;1 SD shown below it); hover any
 cell for mean / median / SD / CV / n / range.</p>
 <div class=meta>
  <span class=chip><b>{len(runs)}</b> runs</span>
  <span class=chip>latest: <b>{latest}</b></span>
  <span class=chip>green = in range</span>
  <span class=chip>&#9650;&#9660; = vs previous run</span>
 </div>
</div>
<div class=card>
 <h2>Runs</h2>
 <p class=note>Cell colour = value vs target; delta colour:
  <span style='color:#5fd35f'>green</span> moved toward target,
  <span style='color:#ff6b6b'>red</span> away.</p>
 <table><tr><th>Date</th><th>Run</th><th>Strokes<br><span class=t>n</span></th>
 {cols}<th>Notes</th><th>Files</th></tr>{''.join(rows)}</table>
 <div class=legend><span style='background:#1e6b2e'>in range</span>
  <span style='background:#8a5a00'>slightly off</span>
  <span style='background:#7a1f1f'>well off</span></div>
</div>
<div class=card>
 <h2>Trends across runs</h2>
 <p class=note>Shaded band = target range &middot; whiskers = &plusmn;1 SD.</p>
 {img}
</div>
</div></body></html>"""


def _dashboard_bg_uri():
    """Embed a background photo (assets/dashboard_bg.* or gui_bg.jpg) as a
    data URI so analysis.html is self-contained. Returns None if none found."""
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("dashboard_bg.jpg", "dashboard_bg.png", "gui_bg.jpg"):
        path = os.path.join(here, "assets", name)
        if os.path.isfile(path):
            mime = "image/png" if name.endswith(".png") else "image/jpeg"
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            return f"data:{mime};base64,{data}"
    return None


def run_analysis(outdir):
    runs = load_runs(outdir)
    if not runs:
        print(f"No runs found in {os.path.abspath(outdir)}. Record some first.")
        return
    png = make_trend_png(runs)
    html = build_dashboard_html(runs, png, _dashboard_bg_uri())
    out = os.path.join(outdir, "analysis.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Analysis dashboard ({len(runs)} runs): {os.path.abspath(out)}")
    webbrowser.open("file://" + os.path.abspath(out))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Side-on bike fit analyzer (IMX415/Windows)")
    ap.add_argument("--source", default="auto",
                    help="'auto' (find IMX415), a camera index (e.g. 0), or a video path")
    ap.add_argument("--side", choices=["left", "right"], default="right",
                    help="which body side faces the camera")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--outdir", default="captures",
                    help="folder for saved video/snapshot/report (default: ./captures)")
    ap.add_argument("--no-save", action="store_true",
                    help="disable all file saving (video, snapshot, report)")
    ap.add_argument("--analyze", action="store_true",
                    help="build the HTML analysis dashboard from saved runs and exit")
    ap.add_argument("--name", default=None,
                    help="run name/label (skips the end-of-run prompt)")
    ap.add_argument("--note", default=None,
                    help="run note (skips the end-of-run prompt)")
    ap.add_argument("--voice", action="store_true",
                    help="hands-free: say 'start'/'stop' to record; spoken replies")
    ap.add_argument("--vosk-model", default=None,
                    help="path to a Vosk model dir (default: ./models/" + VOSK_MODEL_NAME + ")")
    ap.add_argument("--piper-model", default=None,
                    help="path to a Piper .onnx voice (default: ./models/piper/"
                         + PIPER_VOICE_NAME + ".onnx)")
    ap.add_argument("--mic", type=int, default=None,
                    help="input device index for voice (default: system default mic)")
    ap.add_argument("--list-mics", action="store_true",
                    help="list audio input devices and exit")
    args = ap.parse_args()

    if args.list_mics:
        import sounddevice as sd
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                print(f"  [{i}] {d['name']}")
        return

    if args.analyze:
        run_analysis(args.outdir)
        return

    side = L if args.side == "left" else R

    here = os.path.dirname(os.path.abspath(__file__))

    # Each run saves into its own folder: captures/<timestamp>[_<name>]/
    save = not args.no_save
    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = None
    if save:
        os.makedirs(args.outdir, exist_ok=True)
        run_dir = os.path.join(args.outdir, session)
        os.makedirs(run_dir, exist_ok=True)
        print(f"Saving run to: {os.path.abspath(run_dir)}  (--no-save to disable)")

    # Resolve source.
    if args.source == "auto":
        idx = find_imx415()
        source = idx if idx is not None else 0
        if idx is None:
            print("IMX415 not found - falling back to camera index 0")
    elif args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source

    cap = open_camera(source, args.width, args.height)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
    print(f"Running at {aw}x{ah}.  SPACE=record/pause, R=reset capture, Q=quit.")
    print("Start PAUSED. Mount the bike, settle in, then press SPACE to record.")

    # Voice control (optional, hands-free).
    cmd_queue = queue.Queue()
    piper_model = None
    if args.voice:
        piper_model = resolve_piper_model(
            args.piper_model
            or os.path.join(here, "models", "piper", PIPER_VOICE_NAME + ".onnx"))
    speaker = Speaker(enabled=args.voice, piper_model=piper_model)
    if args.voice:
        model_path = args.vosk_model or os.path.join(here, "models", VOSK_MODEL_NAME)
        model_path = resolve_vosk_model(model_path)
        if model_path and start_voice_listener(cmd_queue, model_path, args.mic):
            print("Voice ON: say 'start' to record, 'stop' to finish.")
            speaker.say("Voice control ready")
        else:
            print("Voice OFF: could not start speech recognition; use SPACE.")

    # Display sizing: scale video down to a sane width, panel to the right.
    disp_w = 900
    disp_h = int(disp_w * ah / aw)
    disp_h -= disp_h % 2          # H.264 requires even dimensions
    panel_w = 500

    metric_keys = [k for k, *_ in METRICS]
    captured = None         # running MEDIAN of per-stroke samples (for panel)
    captured_frame = None   # a representative annotated bottom-of-stroke frame
    recording = False       # only capture while actively pedalling
    writer = None           # cv2.VideoWriter, created per recording session
    clip_w, clip_h = disp_w + panel_w, disp_h
    n_clips = 0

    # Per-stroke sampling state. A "stroke" = one crank revolution; its bottom
    # is the frame where the ankle is lowest (max pixel-y). We collect one
    # sample per stroke so we can report mean/median/std across the whole ride.
    strokes = {k: [] for k in metric_keys}
    peak_y = None           # lowest ankle point seen in the current cycle
    peak_angles = None      # angles snapshot at that bottom
    fallback = None         # deepest-knee angles, used if no full stroke seen
    fallback_knee = -1.0

    def begin_recording():
        nonlocal recording, captured, captured_frame, writer, n_clips
        nonlocal peak_y, peak_angles, fallback, fallback_knee
        if recording:
            return
        # Fresh session: clear any previous capture so only this clean
        # pedalling stretch counts.
        recording = True
        captured = None
        captured_frame = None
        peak_y = None
        peak_angles = None
        fallback = None
        fallback_knee = -1.0
        for k in metric_keys:
            strokes[k] = []
        if save:
            n_clips += 1
            clip_path = os.path.join(run_dir, f"clip{n_clips}.mp4")
            writer = open_video_writer(clip_path, 30.0, (clip_w, clip_h))
            print(f"Recording STARTED -> {clip_path}")
        else:
            print("Recording STARTED - pedal steadily.")
        speaker.say("Recording started")

    def end_recording():
        nonlocal recording, writer
        if not recording:
            return
        recording = False
        if writer is not None:
            writer.release()
            writer = None
        print("Recording PAUSED - capture frozen.")
        speaker.say("Recording stopped")

    with mp_pose.Pose(model_complexity=1, min_detection_confidence=0.6,
                      min_tracking_confidence=0.6, smooth_landmarks=True) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed (end of clip or camera dropped).")
                break
            h, w = frame.shape[:2]
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            live = None
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                p = lambda name: px(lm, side[name], w, h)
                live = compute_angles(p)

                # Per-stroke sampling (only while recording, so mounting /
                # dismounting can't poison the data).
                stroke_done = False
                if recording:
                    ankle_y = p("ankle")[1]
                    cur = {k: live[k] for k in metric_keys}
                    if live["knee_ext"] > fallback_knee:   # deepest-knee fallback
                        fallback_knee = live["knee_ext"]
                        fallback = cur
                    drop = 0.04 * h                        # hysteresis (~4% height)
                    if peak_y is None or ankle_y >= peak_y:
                        peak_y = ankle_y                   # still descending to bottom
                        peak_angles = cur
                    elif peak_angles is not None and ankle_y < peak_y - drop:
                        for k in metric_keys:              # foot back up -> one stroke
                            strokes[k].append(peak_angles[k])
                        captured = {k: float(np.median(strokes[k]))
                                    for k in metric_keys}
                        stroke_done = True
                        peak_y = ankle_y
                        peak_angles = None

                draw_skeleton(frame, res.pose_landmarks, side, live)
                # live angle labels at the relevant joints
                p_int = lambda name: tuple(map(int, p(name)))
                cv2.putText(frame, f"Knee ext {live['knee_ext']:.0f}",
                            (p_int("knee")[0] + 10, p_int("knee")[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2)
                cv2.putText(frame, f"Hip {live['hip_open']:.0f}",
                            (p_int("hip")[0] + 10, p_int("hip")[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2)
                cv2.putText(frame, f"Shldr {live['shoulder']:.0f}",
                            (p_int("shoulder")[0] + 10, p_int("shoulder")[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2)
                cv2.putText(frame, f"Arm {live['arm']:.0f}",
                            (p_int("elbow")[0] + 10, p_int("elbow")[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2)

                # Keep a representative annotated bottom-of-stroke frame: prefer
                # an actual confirmed stroke; otherwise fall back to the deepest.
                if stroke_done or (captured_frame is None and recording
                                   and fallback is not None):
                    captured_frame = frame.copy()
            else:
                cv2.putText(frame, "No pose detected - step into frame, side-on",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 2)

            n_strokes = len(strokes[metric_keys[0]])
            video = cv2.resize(frame, (disp_w, disp_h))
            if recording:
                cv2.circle(video, (28, 30), 10, RED, -1)
                cv2.putText(video, f"REC   strokes: {n_strokes}", (46, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)
            else:
                cv2.putText(video, "PAUSED  (SPACE to record)", (46, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREY, 2)
            panel = render_panel(panel_w, disp_h, captured, live, recording,
                                 n_strokes)
            composite = np.hstack([video, panel])
            cv2.imshow("BikePositionCV - side-on fit", composite)

            # Write the annotated video while recording.
            if recording and writer is not None:
                writer.write(composite)

            # Voice commands (drain anything the listener heard this frame).
            # Always answer out loud so you know you were heard, even if the
            # state didn't change.
            try:
                while True:
                    cmd = cmd_queue.get_nowait()
                    if cmd == "start":
                        if recording:
                            speaker.say("Already recording")
                        else:
                            begin_recording()
                    elif cmd == "stop":
                        if recording:
                            end_recording()
                        else:
                            speaker.say("Not recording yet")
            except queue.Empty:
                pass

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                end_recording() if recording else begin_recording()
            elif key == ord("r"):
                captured = None
                captured_frame = None
                peak_y = None
                peak_angles = None
                fallback = None
                fallback_knee = -1.0
                for k in metric_keys:
                    strokes[k] = []
                print("Capture reset - pedal a full stroke to re-capture.")

    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()

    # Aggregate per-stroke samples into descriptive statistics.
    if any(strokes[k] for k in metric_keys):
        stats = {k: metric_stats(strokes[k]) for k in metric_keys}
        captured = {k: (stats[k]["median"] if stats[k] else None)
                    for k in metric_keys}
    else:
        stats = None
        captured = fallback   # single deepest frame (or None if nothing seen)

    # Save snapshot + report into the run folder.
    txt_path = csv_path = None
    if save:
        if captured_frame is not None:
            cv2.imwrite(os.path.join(run_dir, "bottom.jpg"), captured_frame)
            print(f"  Snapshot saved: {os.path.join(run_dir, 'bottom.jpg')}")
        txt_path = os.path.join(run_dir, "report.txt")
        csv_path = os.path.join(run_dir, "report.csv")
    fit_report(captured, stats, txt_path, csv_path)

    # Name + note this run, write the structured record, tidy the folder name.
    if save:
        name, note = args.name, args.note
        try:
            if name is None:
                name = input("\nRun name (e.g. '5mm saddle lowered'): ").strip()
            if note is None:
                note = input("Notes (optional): ").strip()
        except EOFError:
            name, note = name or "", note or ""
        write_run_json(run_dir, session, name, note, args.side, captured, stats)
        if name:
            new_dir = os.path.join(args.outdir, f"{session}_{slugify(name)}")
            try:
                if os.path.abspath(new_dir) != os.path.abspath(run_dir):
                    os.rename(run_dir, new_dir)
                    run_dir = new_dir
            except OSError:
                pass
        print(f"Saved run -> {os.path.abspath(run_dir)}")
        print("View all runs:  python BikePositionCV.py --analyze")


if __name__ == "__main__":
    main()
