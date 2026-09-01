import time, sys
import numpy as np, cv2, mediapipe as mp
from phase1_landmark_view import CameraThread, build_landmarker

lm = build_landmarker()
for (w, h) in [(1280, 720), (640, 480)]:
    cam = CameraThread(width=w, height=h)
    time.sleep(1.0)
    infer, seen, total, last, idx = [], 0, 0, 0.0, 0
    end = time.perf_counter() + 6.0
    while time.perf_counter() < end:
        frame, stamp = cam.read()
        if frame is None or stamp == last:
            time.sleep(0.003); continue
        last = stamp
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                  int(idx * 1000 / 30) + (0 if w == 1280 else 100000))
        infer.append((time.perf_counter() - t0) * 1000); idx += 1; total += 1
        seen += bool(res.hand_landmarks)
    aw, ah, _, fc = cam.settings()
    cam.release()
    a = np.array(infer[10:])   # drop warmup
    print(f"{aw}x{ah} {fc}: cam {cam.fps:4.1f}fps  infer mean {a.mean():5.1f}ms  "
          f"p50 {np.percentile(a,50):5.1f}  p95 {np.percentile(a,95):5.1f}  seen {100*seen/total:.0f}%")
lm.close()
