"""Headless check: camera opens, delivers frames, and inference runs.
No window - just proves the pipeline before running the interactive viewer.
"""
import time
import numpy as np
from phase1_landmark_view import CameraThread, build_landmarker, ensure_model
import cv2, mediapipe as mp

ensure_model()
cam = CameraThread()
w, h, fps, fourcc = cam.settings()
print(f"negotiated: {w}x{h} @ {fps:.0f}fps {fourcc}")
time.sleep(1.0)

lm = build_landmarker()
infer, seen, total, last, idx = [], 0, 0, 0.0, 0
end = time.perf_counter() + 5.0
while time.perf_counter() < end:
    frame, stamp = cam.read()
    if frame is None or stamp == last:
        time.sleep(0.003); continue
    last = stamp
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    t0 = time.perf_counter()
    res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                              int(idx * 1000 / 30))
    infer.append((time.perf_counter() - t0) * 1000)
    idx += 1; total += 1
    seen += bool(res.hand_landmarks)

cam.release(); lm.close()
a = np.array(infer)
print(f"frames processed: {total}")
print(f"camera fps:       {cam.fps:.1f}")
print(f"inference:        mean {a.mean():.1f}ms  p95 {np.percentile(a,95):.1f}ms")
print(f"hand seen in:     {100*seen/max(total,1):.0f}% of frames")
