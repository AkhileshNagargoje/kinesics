"""Phase 3b: thumbs up / thumbs down as scroll direction.

Replaces the drag design. Direction comes from WHICH pose you make, not from
how your hand moves, and the hand holds still while scrolling.

That is what fixes the bug the drag version kept hitting: downward strokes were
~1.5x faster than upward ones, fast motion blurred the frame, MediaPipe lost
the hand, and the gate dropped mid-stroke (5 releases down vs 2 up). A still
hand cannot blur.

What is given up is analog speed - a pose is on or off. So speed instead comes
from how long you hold it: slow at first for precise nudges, accelerating to a
fast travel speed, the way holding an arrow key behaves.

    thumb up   + other fingers curled -> scroll up
    thumb down + other fingers curled -> scroll down

Starts DISARMED. Keys: a arm | [ ] speed | r ramp on/off | q quit
"""

import sys
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from enroll import analyse
from phase1_landmark_view import CameraThread, build_landmarker, ensure_model
from phase2_pose_detect import FRAMES_TO_ENGAGE, MIN_HAND_SIZE_PX
from phase3_scroll import Wheel, _anchor
from trace_pose import margins

# Releasing takes longer than engaging: a still hand still drops out of
# tracking occasionally, and a dropped frame should not chop the scroll.
FRAMES_TO_RELEASE = 10

THUMB_MIN = 0.55        # cosine of thumb direction vs vertical; 1.0 = straight up
FINGER_CURL_MAX = 0.05  # the other four must be curled, with a little slack

BASE_RATE = 260.0       # wheel units/sec at the start of a hold
MAX_RATE = 1400.0       # after ramping up
RAMP_TIME = 1.6         # seconds to reach full speed
ONSET_DELAY = 0.12      # ignore the first moment, while the pose settles


def thumb_state(pts):
    """Return +1 for thumbs up, -1 for thumbs down, 0 for neither.

    Requires the other four fingers curled and the thumb actually extended,
    then reads the direction the thumb points.
    """
    size = float(np.linalg.norm(pts[9] - pts[0]))
    if size <= 1e-6:
        return 0, 0.0

    m = margins(pts)
    if any(m[f] > FINGER_CURL_MAX for f in ("index", "middle", "ring", "pinky")):
        return 0, 0.0

    # thumb must actually be sticking out, not tucked into the fist
    thumb_out = (np.linalg.norm(pts[4] - pts[0]) - np.linalg.norm(pts[3] - pts[0])) / size
    if thumb_out < 0.05:
        return 0, 0.0

    # Which way the thumb POINTS, not where it sits. A relaxed fist rests with
    # the thumb slightly below the knuckle line, which a position test reads as
    # a weak thumbs-down; the pointing direction of that same thumb is nearly
    # sideways, so it is correctly rejected.
    v = pts[4] - pts[2]                                # thumb MCP -> tip
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return 0, 0.0
    reach = float(-v[1] / n)                           # +1 straight up, -1 straight down
    if reach > THUMB_MIN:
        return 1, reach
    if reach < -THUMB_MIN:
        return -1, reach
    return 0, reach


class ThumbScroller:
    # Swappable so a different gesture can reuse the ramp and hysteresis.
    detect = staticmethod(thumb_state)

    def __init__(self):
        self.wheel = Wheel()
        self.direction = 0          # current engaged direction
        self.candidate = 0
        self.true_run = 0
        self.false_run = 0
        self.held_since = 0.0
        self.armed = False
        self.ramp = True
        self.base = BASE_RATE
        self.last_emit = 0.0
        self.total = 0.0
        self.holds = 0
        self.reach = 0.0

    def rate(self, now):
        held = now - self.held_since - ONSET_DELAY
        if held <= 0:
            return 0.0
        if not self.ramp:
            return self.base
        t = min(1.0, held / RAMP_TIME)
        # ease-in: gentle at the start where precision matters, fast later
        return self.base + (MAX_RATE - self.base) * (t * t)

    def update(self, state, dt, now):
        d, reach = (0, 0.0)
        if state is not None and state.size >= MIN_HAND_SIZE_PX:
            d, reach = self.detect(state.pts)
        self.reach = reach

        if d != 0 and (self.direction == 0 or d == self.direction):
            self.true_run += 1
            self.false_run = 0
            self.candidate = d
        else:
            self.false_run += 1
            self.true_run = 0

        if self.direction == 0 and self.true_run >= FRAMES_TO_ENGAGE:
            self.direction = self.candidate
            self.held_since = now
            self.holds += 1
        elif self.direction != 0 and self.false_run >= FRAMES_TO_RELEASE:
            self.direction = 0

        emitted = 0.0
        if self.direction:
            # image y grows downward; thumbs up should scroll the page up,
            # which is a POSITIVE wheel delta
            emitted = self.direction * self.rate(now) * dt
            if self.armed:
                self.wheel.scroll(emitted)
            self.total += abs(emitted)

        self.last_emit = emitted
        return emitted


def hud(canvas, sc, state, now, fps):
    h, w = canvas.shape[:2]
    if not sc.armed:
        label, color = "DISARMED  (press a)", (110, 110, 240)
    elif sc.direction > 0:
        label, color = "SCROLL UP", (110, 235, 110)
    elif sc.direction < 0:
        label, color = "SCROLL DOWN", (110, 235, 110)
    else:
        label, color = "armed - thumbs up or down", (0, 190, 240)

    cv2.rectangle(canvas, (0, 0), (w, 46), (0, 0, 0), -1)
    cv2.putText(canvas, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    if sc.direction and sc.armed:
        cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), color, 5)

    # speed ramp gauge
    cx, cy = w - 44, h // 2
    cv2.rectangle(canvas, (cx - 9, cy - 70), (cx + 9, cy + 70), (70, 70, 70), 2)
    if sc.direction:
        frac = np.clip((sc.rate(now) - sc.base) / max(MAX_RATE - sc.base, 1e-6), 0, 1)
        top = int(cy + 70 - 140 * frac)
        cv2.rectangle(canvas, (cx - 7, top), (cx + 7, cy + 70), (110, 235, 110), -1)

    for i, s in enumerate([
        f"thumb reach {sc.reach:+.2f}  (need +-{THUMB_MIN:.2f})   "
        f"rate {sc.rate(now):.0f}/s   ramp {'on' if sc.ramp else 'off'}",
        f"holds {sc.holds}   emitted {sc.total:.0f} units   {fps:.0f} fps",
        "a arm   [ ] base speed   r ramp   q quit",
    ]):
        y = h - 52 + i * 18
        cv2.putText(canvas, s, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, s, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (185, 185, 185), 1, cv2.LINE_AA)


def main():
    ensure_model()
    cam = CameraThread()
    landmarker = build_landmarker()
    sc = ThumbScroller()
    big = "--big" in sys.argv

    last_stamp = 0.0
    frame_index = 0
    fps_hist = deque(maxlen=30)
    lost_during = held_frames = 0

    print("phase 3b running - DISARMED. Focus the window and press 'a' to arm.")
    try:
        while True:
            frame, stamp = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue
            if stamp == last_stamp:
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue
            dt = min(stamp - last_stamp, 0.2) if last_stamp else 1 / 30
            last_stamp = stamp
            fps_hist.append(1.0 / max(dt, 1e-6))
            now = time.perf_counter()

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                int(frame_index * 1000 / 30))
            frame_index += 1

            h, w = frame.shape[:2]
            state = None
            if res.hand_landmarks:
                pts = np.array([[p.x * w, p.y * h] for p in res.hand_landmarks[0]])
                state = analyse(pts, res.handedness[0][0].category_name)

            sc.update(state, dt, now)
            if sc.direction:
                held_frames += 1
                lost_during += state is None

            canvas = frame if big else cv2.resize(frame, (640, 360))
            if state is not None:
                s = canvas.shape[1] / w
                for a, b in ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 9),
                             (9, 13), (13, 17), (0, 17)):
                    cv2.line(canvas, tuple((state.pts[a] * s).astype(int)),
                             tuple((state.pts[b] * s).astype(int)),
                             (90, 200, 120), 2, cv2.LINE_AA)
                cv2.circle(canvas, tuple((state.pts[4] * s).astype(int)), 9,
                           (60, 90, 255), -1, cv2.LINE_AA)

            hud(canvas, sc, state, now, np.mean(fps_hist))
            cv2.imshow("phase 3b - thumb scroll", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("a"):
                sc.armed = not sc.armed
                print("ARMED" if sc.armed else "disarmed")
            elif key == ord("r"):
                sc.ramp = not sc.ramp
            elif key == ord("]"):
                sc.base *= 1.25
            elif key == ord("["):
                sc.base /= 1.25
    finally:
        cam.release()
        landmarker.close()
        cv2.destroyAllWindows()

    print(f"\n{sc.holds} holds, {sc.total:.0f} wheel units ({sc.total / 120:.0f} notches)")
    print(f"final base rate {sc.base:.0f}/s, ramp {'on' if sc.ramp else 'off'}")
    if held_frames:
        print(f"hand lost during {100 * lost_during / held_frames:.1f}% of scrolling frames"
              f"  (was 40.9% with the drag design)")


if __name__ == "__main__":
    main()
