"""Record a timed measurement run and report the numbers phase 3 needs.

Unlike the viewer, this writes data instead of pixels. Run it once per
condition so the conditions can be compared honestly.

    python measure.py normal-light
    python measure.py backlit
    python measure.py dim
    python measure.py far

Hold the two-finger pose AS STILL AS YOU CAN for the whole run. Stillness is
the point: any anchor movement recorded here is tracker noise, and that noise
is what sets the smoothing constants and the deadzone.

Reports:
  cam fps / infer ms   throughput and headroom
  seen %               how often the hand was found at all
  size                 scale reference, and how much it drifts
  jitter               px std-dev of the anchor while your hand is still,
                       raw and normalized by hand size
  worst jump           largest single-frame anchor movement - this is what
                       sets the deadzone, since anything smaller than it
                       would fire scrolls from noise alone
"""

import csv
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from phase1_landmark_view import CameraThread, build_landmarker, ensure_model, hand_size

DURATION = 10.0
OUT_DIR = Path(__file__).parent / "measurements"


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "unlabeled"
    ensure_model()
    OUT_DIR.mkdir(exist_ok=True)

    cam = CameraThread()
    w, h, _, fourcc = cam.settings()
    landmarker = build_landmarker()
    time.sleep(1.0)                      # let exposure settle before measuring

    print(f"[{label}] {w}x{h} {fourcc} - hold the pose still for {DURATION:.0f}s")
    for n in (3, 2, 1):
        print(f"  {n}...")
        time.sleep(1.0)
    print("  recording")

    rows = []
    last, idx = 0.0, 0
    end = time.perf_counter() + DURATION
    while time.perf_counter() < end:
        frame, stamp = cam.read()
        if frame is None or stamp == last:
            time.sleep(0.002)
            continue
        last = stamp

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        res = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(idx * 1000 / 30)
        )
        infer = (time.perf_counter() - t0) * 1000
        idx += 1

        if res.hand_landmarks:
            pts = np.array([[p.x * w, p.y * h] for p in res.hand_landmarks[0]])
            anchor = (pts[5] + pts[9]) / 2.0
            rows.append(dict(t=stamp, infer_ms=infer, seen=1,
                             ax=anchor[0], ay=anchor[1], size=hand_size(pts)))
        else:
            rows.append(dict(t=stamp, infer_ms=infer, seen=0,
                             ax="", ay="", size=""))

    cam.release()
    landmarker.close()

    path = OUT_DIR / f"{label}.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t", "infer_ms", "seen", "ax", "ay", "size"])
        writer.writeheader()
        writer.writerows(rows)

    report(label, rows, cam.fps, path)


def report(label, rows, cam_fps, path):
    total = len(rows)
    hits = [r for r in rows if r["seen"] == 1]
    infer = np.array([r["infer_ms"] for r in rows])

    print(f"\n=== {label} ===")
    print(f"frames        {total}   cam {cam_fps:.1f} fps")
    print(f"infer         mean {infer.mean():.1f}ms   p95 {np.percentile(infer, 95):.1f}ms")
    print(f"hand seen     {100 * len(hits) / max(total, 1):.1f}%")

    if len(hits) < 30:
        print("not enough detected frames to measure jitter")
        return

    ax = np.array([r["ax"] for r in hits])
    ay = np.array([r["ay"] for r in hits])
    size = np.array([r["size"] for r in hits])

    # Jitter measured against a slow-moving baseline rather than the overall
    # mean, so a gentle hand drift over 10s is not counted as noise.
    k = 15
    kernel = np.ones(k) / k
    bx = np.convolve(ax, kernel, mode="same")
    by = np.convolve(ay, kernel, mode="same")
    resid = np.hypot(ax - bx, ay - by)[k:-k]

    steps = np.hypot(np.diff(ax), np.diff(ay))

    print(f"size          mean {size.mean():.1f}px  (normalized {size.mean() / 720:.3f})"
          f"  drift +-{size.std():.1f}px")
    print(f"jitter        rms {resid.mean():.2f}px   p95 {np.percentile(resid, 95):.2f}px"
          f"   = {100 * resid.mean() / size.mean():.2f}% of hand size")
    print(f"frame steps   median {np.median(steps):.2f}px   p95 {np.percentile(steps, 95):.2f}px"
          f"   max {steps.max():.2f}px")
    print(f"\nsuggested deadzone: {np.percentile(steps, 99):.1f}px per frame "
          f"({100 * np.percentile(steps, 99) / size.mean():.1f}% of hand size)")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
