"""
bikefit.py — DIY side-on bike fit analyzer
==========================================

Markerless bike-fit tool in the spirit of MyVeloFit. Runs MediaPipe Pose on a
side-on view of you pedalling (live from your IMX415 USB cam, or a recorded
clip) and tracks the four angles that matter for road position:

    - knee angle      (hip-knee-ankle)      -> SADDLE HEIGHT
    - torso/back angle (hip->shoulder vs horizontal) -> REACH / BAR DROP
    - shoulder angle  (hip-shoulder-elbow)   -> REACH
    - elbow angle     (shoulder-elbow-wrist) -> REACH / COMFORT

It overlays the angles live and, on quit, prints a report comparing your
measured angles against reference ranges with adjustment hints.

SETUP
-----
    pip install opencv-python mediapipe numpy

CAMERA RIG (this matters more than the model)
    - Tripod, lens height ~ bottom-bracket height.
    - Camera axis exactly perpendicular to the bike's plane.
    - Stand BACK 3-4 m and let the 4K sensor frame it. Don't get close with a
      wide lens -> barrel distortion bends limbs and corrupts angles.
    - Good, even lighting; rider in contrasting clothing helps.

USAGE
    # live, drive(right) side toward camera:
    python bikefit.py --source 0 --side right
    # analyse a recorded clip:
    python bikefit.py --source ride.mp4 --side left
    Press 'q' to stop and print the report.
"""

import argparse
import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# MediaPipe Pose landmark indices
L = dict(shoulder=11, elbow=13, wrist=15, hip=23, knee=25, ankle=27)
R = dict(shoulder=12, elbow=14, wrist=16, hip=24, knee=26, ankle=28)

# Reference windows (road, endurance-ish). Treat as starting points, not gospel.
# knee_ext = MAX open knee angle over the session (~bottom of stroke).
TARGETS = {
    "knee_ext":  (140, 150, "Saddle height"),
    "torso":     (40,  50,  "Reach / bar drop"),
    "shoulder":  (80,  90,  "Reach"),
    "elbow":     (150, 165, "Reach / arm comfort"),
}


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


def fit_report(stats):
    print("\n" + "=" * 56)
    print(" BIKE FIT REPORT  (measured vs reference window)")
    print("=" * 56)
    for key, (lo, hi, what) in TARGETS.items():
        val = stats.get(key)
        if val is None:
            print(f"  {key:9s}: no data")
            continue
        if val < lo:
            verdict, hint = "LOW ", low_hint(key)
        elif val > hi:
            verdict, hint = "HIGH", high_hint(key)
        else:
            verdict, hint = "OK  ", ""
        print(f"  {key:9s}: {val:6.1f}deg   target {lo}-{hi}   [{verdict}] "
              f"-> {what}{('  | ' + hint) if hint else ''}")
    print("=" * 56)
    print("  Note: saddle TILT can't be read from a side-on angle tool —")
    print("  set it near level and tune by feel/pressure.\n")


def low_hint(key):
    return {
        "knee_ext": "knee too bent at bottom -> RAISE saddle",
        "torso":    "torso too low/stretched -> shorten reach (shorter stem / raise bars)",
        "shoulder": "reaching too far -> shorten reach",
        "elbow":    "arms too straight/locked -> shorten reach or bend elbows",
    }[key]


def high_hint(key):
    return {
        "knee_ext": "leg over-extended / hips may rock -> LOWER saddle",
        "torso":    "torso quite upright -> can drop bars / longer stem if you want lower",
        "shoulder": "very closed shoulder -> lengthen reach",
        "elbow":    "arms very bent -> lengthen reach",
    }[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0",
                    help="camera index (e.g. 0) or path to a video file")
    ap.add_argument("--side", choices=["left", "right"], default="right",
                    help="which body side faces the camera")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    side = L if args.side == "left" else R
    src = int(args.source) if args.source.isdigit() else args.source

    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    # Running aggregates across the session
    knee_max = -1.0          # max open knee angle == extension at bottom of stroke
    knee_min = 999.0
    torso_vals, sh_vals, el_vals = [], [], []

    with mp_pose.Pose(model_complexity=1, min_detection_confidence=0.6,
                      min_tracking_confidence=0.6, smooth_landmarks=True) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                p = lambda name: px(lm, side[name], w, h)
                hip, knee, ankle = p("hip"), p("knee"), p("ankle")
                sh, elb, wr = p("shoulder"), p("elbow"), p("wrist")

                knee_a  = angle_3pt(hip, knee, ankle)
                torso_a = angle_to_horizontal(hip, sh)
                sh_a    = angle_3pt(hip, sh, elb)
                el_a    = angle_3pt(sh, elb, wr)

                knee_max = max(knee_max, knee_a)
                knee_min = min(knee_min, knee_a)
                torso_vals.append(torso_a); sh_vals.append(sh_a); el_vals.append(el_a)

                mp_draw.draw_landmarks(frame, res.pose_landmarks,
                                       mp_pose.POSE_CONNECTIONS)
                for label, val, pt in [
                    (f"knee {knee_a:.0f}", knee_a, knee),
                    (f"torso {torso_a:.0f}", torso_a, hip),
                    (f"shldr {sh_a:.0f}", sh_a, sh),
                    (f"elbow {el_a:.0f}", el_a, elb),
                ]:
                    cv2.putText(frame, label, (int(pt[0]) + 8, int(pt[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"knee ext (max): {knee_max:.0f}  flex (min): {knee_min:.0f}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("bikefit  (q to finish)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    # Use medians for the reach angles (robust to jitter); max knee for extension.
    stats = {
        "knee_ext": knee_max if knee_max > 0 else None,
        "torso":    float(np.median(torso_vals)) if torso_vals else None,
        "shoulder": float(np.median(sh_vals)) if sh_vals else None,
        "elbow":    float(np.median(el_vals)) if el_vals else None,
    }
    fit_report(stats)


if __name__ == "__main__":
    main()