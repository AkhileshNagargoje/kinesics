"""Phase 1: webcam feed + hand landmarks + performance readout.

Diagnostic, not functional. It answers one question: are this camera and this
lighting good enough to build gesture scrolling on top of?

Three numbers matter:
  cam    frames/sec the webcam actually delivers. Should sit near 30. If it
         sags toward 15 in dim light that is auto-exposure lengthening the
         shutter, and it will cost responsiveness in phase 3.
  infer  milliseconds per landmark inference. Expect 5-10ms on this CPU.
  size   hand size in normalized units. Later this is what makes scroll gain
         independent of how far you sit from the camera.

Keys: q quit | h mirror | b black background (landmarks only)
"""

import sys
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Standard 21-point hand topology, hardcoded rather than imported so a library
# reshuffle cannot silently change what gets drawn.
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (9, 10), (10, 11), (11, 12),               # middle
    (13, 14), (14, 15), (15, 16),              # ring
    (0, 17), (17, 18), (18, 19), (19, 20),     # pinky
    (5, 9), (9, 13), (13, 17),                 # palm
]
FINGERTIPS = {4, 8, 12, 16, 20}


class CameraThread:
    """Grabs frames continuously, keeps only the newest.

    Inference is slower than capture, so without this the driver's buffer fills
    and you process stale frames - latency that grows the longer you run.
    Dropping frames here keeps it flat.
    """

    def __init__(self, index=0, width=1280, height=720, fps=30):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open camera {index}")
        # MJPG first: many webcams offer 30fps at 720p only in MJPG and drop to
        # ~10fps raw YUY2 otherwise.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.frame = None
        self.stamp = 0.0
        self.lock = threading.Lock()
        self.running = True
        self.intervals = deque(maxlen=60)

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        prev = time.perf_counter()
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            now = time.perf_counter()
            self.intervals.append(now - prev)
            prev = now
            with self.lock:
                self.frame = frame
                self.stamp = now

    def read(self):
        with self.lock:
            if self.frame is None:
                return None, 0.0
            return self.frame.copy(), self.stamp

    @property
    def fps(self):
        if not self.intervals:
            return 0.0
        mean = sum(self.intervals) / len(self.intervals)
        return 1.0 / mean if mean > 0 else 0.0

    def settings(self):
        code = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((code >> 8 * i) & 0xFF) for i in range(4))
        return (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            self.cap.get(cv2.CAP_PROP_FPS),
            fourcc,
        )

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


def ensure_model():
    if MODEL_PATH.exists():
        return
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading hand_landmarker.task (~7.5 MB)\n  from {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"saved to {MODEL_PATH}")


def build_landmarker():
    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(opts)


def hand_size(pts):
    """Wrist -> middle-finger MCP distance in pixels.

    The scale reference everything later divides by, so leaning toward the
    camera does not change how far a given hand motion scrolls.
    """
    return float(np.linalg.norm(pts[9] - pts[0]))


def draw_hand(canvas, pts):
    # pts stays float for the scale math; drawing needs int pixel coords.
    px = [(int(round(x)), int(round(y))) for x, y in pts]
    for a, b in CONNECTIONS:
        cv2.line(canvas, px[a], px[b], (90, 220, 120), 2, cv2.LINE_AA)
    for i, p in enumerate(px):
        tip = i in FINGERTIPS
        cv2.circle(canvas, p, 6 if tip else 4,
                   (60, 90, 255) if tip else (240, 240, 240), -1, cv2.LINE_AA)
    # Anchor: midpoint of the index and middle knuckles. Stable under finger
    # flexion, unlike the fingertips - this is the point phase 3 will track.
    mid = (pts[5] + pts[9]) / 2.0
    cv2.circle(canvas, (int(round(mid[0])), int(round(mid[1]))), 9,
               (0, 215, 255), 2, cv2.LINE_AA)


def put(canvas, text, y, color=(235, 235, 235)):
    cv2.putText(canvas, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def main():
    ensure_model()
    cam = CameraThread(index=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    w, h, fps, fourcc = cam.settings()
    print(f"camera negotiated: {w}x{h} @ {fps:.0f}fps {fourcc}")

    landmarker = build_landmarker()
    infer_ms = deque(maxlen=60)
    mirror = True
    blank_bg = False
    last_stamp = 0.0
    frame_index = 0
    detected = 0
    total = 0

    try:
        while True:
            frame, stamp = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue
            if stamp == last_stamp:      # no new frame yet, do not re-infer
                cv2.waitKey(1)
                continue
            last_stamp = stamp

            if mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            t0 = time.perf_counter()
            # VIDEO mode needs a monotonically increasing timestamp; it uses it
            # to decide when tracking can be reused instead of re-detecting.
            result = landmarker.detect_for_video(mp_image, int(frame_index * 1000 / 30))
            infer_ms.append((time.perf_counter() - t0) * 1000)
            frame_index += 1
            total += 1

            canvas = np.zeros_like(frame) if blank_bg else frame
            fh, fw = canvas.shape[:2]

            size = 0.0
            handed = "-"
            if result.hand_landmarks:
                detected += 1
                lm = result.hand_landmarks[0]
                pts = np.array([[p.x * fw, p.y * fh] for p in lm])
                draw_hand(canvas, pts)
                size = hand_size(pts) / fh          # normalized by frame height
                handed = result.handedness[0][0].category_name

            mean_infer = sum(infer_ms) / len(infer_ms) if infer_ms else 0.0
            put(canvas, f"cam {cam.fps:5.1f} fps   infer {mean_infer:5.1f} ms", 28)
            put(canvas,
                f"hand {handed:<5} size {size:.3f}   seen {100.0 * detected / max(total, 1):5.1f}%",
                52,
                (120, 235, 120) if result.hand_landmarks else (120, 120, 235))
            put(canvas, "q quit   h mirror   b background", fh - 16, (170, 170, 170))

            cv2.imshow("phase 1 - landmark view", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("h"):
                mirror = not mirror
            if key == ord("b"):
                blank_bg = not blank_bg
    finally:
        cam.release()
        landmarker.close()
        cv2.destroyAllWindows()
        if infer_ms:
            arr = np.array(infer_ms)
            print(f"inference: mean {arr.mean():.1f}ms  p95 {np.percentile(arr, 95):.1f}ms")
        print(f"camera delivered {cam.fps:.1f} fps; hand seen in "
              f"{100.0 * detected / max(total, 1):.1f}% of frames")


if __name__ == "__main__":
    main()
