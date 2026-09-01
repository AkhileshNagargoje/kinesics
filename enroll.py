"""Guided hand enrollment - Android face-unlock style, for hands.

Walks you through a sequence of prompts ("tilt left", "move closer", "other
hand"). Each step shows a ring that fills while you hold the requested
position and ticks when the step is satisfied. When every step is ticked the
run is COMPLETE and it exits on its own - no guessing whether it worked, and
no window left running forever.

It does two jobs at once:
  1. Verification. If the tracker is locked onto something that is not your
     hand, the steps will not complete - a false detection cannot tilt left
     on command. Completing the sequence is proof the tracking is real.
  2. Recording. Every frame is written to recordings/, tagged with the step,
     so the enrollment IS the variation dataset discussed earlier.

    python enroll.py                 # writes recordings/session-01.csv
    python enroll.py evening-lamp    # label the lighting condition

Keys: q abort | s skip current step (recorded as skipped)
"""

import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import mediapipe as mp
import numpy as np

from phase1_landmark_view import (
    CameraThread,
    build_landmarker,
    draw_hand,
    ensure_model,
    hand_size,
)

REC_DIR = Path(__file__).parent / "recordings"
HOLD_SECONDS = 1.2          # how long a position must be held to count
DECAY = 2.0                 # ring drains this many times faster than it fills

TIP = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIP = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}


# --------------------------------------------------------------------------
# hand feature extraction
# --------------------------------------------------------------------------

@dataclass
class HandState:
    pts: np.ndarray
    size: float             # wrist -> middle MCP, px
    angle: float            # degrees; 0 = fingers up, + = tilted right
    spread: float           # index-middle tip gap, as a fraction of hand size
    two_fingers: bool
    handed: str


def extended(pts, name):
    """Fingertip farther from the wrist than that finger's PIP joint."""
    wrist = pts[0]
    return np.linalg.norm(pts[TIP[name]] - wrist) > np.linalg.norm(pts[PIP[name]] - wrist)


def analyse(pts, handed) -> HandState:
    size = hand_size(pts)
    v = pts[9] - pts[0]                      # wrist -> middle knuckle
    angle = float(np.degrees(np.arctan2(v[0], -v[1])))
    spread = float(np.linalg.norm(pts[TIP["index"]] - pts[TIP["middle"]]) / max(size, 1e-6))
    two = (extended(pts, "index") and extended(pts, "middle")
           and not extended(pts, "ring") and not extended(pts, "pinky"))
    return HandState(pts, size, angle, spread, two, handed)


# --------------------------------------------------------------------------
# steps
# --------------------------------------------------------------------------

@dataclass
class Step:
    key: str
    prompt: str
    hint: str
    test: Callable[[HandState, dict], bool]
    progress: float = 0.0
    done: bool = False
    skipped: bool = False
    frames: int = 0
    samples: list = field(default_factory=list)


def build_steps():
    """Ordered prompts. Later steps compare against the baseline captured in
    step 1, so distance thresholds adapt to wherever you naturally hold your
    hand rather than to a hardcoded pixel size."""
    return [
        Step("upright", "Hold two fingers up",
             "index and middle up, ring and pinky curled, facing the camera",
             lambda s, b: s.two_fingers and abs(s.angle) < 18),
        Step("tilt_left", "Tilt your hand LEFT",
             "keep the two fingers up, lean the whole hand over",
             lambda s, b: s.two_fingers and s.angle < -28),
        Step("tilt_right", "Tilt your hand RIGHT",
             "same again, the other way",
             lambda s, b: s.two_fingers and s.angle > 28),
        Step("near", "Move your hand CLOSER",
             "toward the camera, fingers still up",
             lambda s, b: s.two_fingers and s.size > b["size"] * 1.30),
        Step("far", "Move your hand BACK",
             "arm out, as far as you would still scroll from",
             lambda s, b: s.two_fingers and s.size < b["size"] * 0.78),
        Step("spread", "SPREAD the two fingers apart",
             "wide V",
             lambda s, b: s.two_fingers and s.spread > 0.62),
        Step("together", "Close the two fingers TOGETHER",
             "fingers touching, still both extended",
             lambda s, b: s.two_fingers and s.spread < 0.34),
        Step("other_hand", "Now your OTHER hand",
             "same two-finger pose",
             lambda s, b: s.two_fingers and s.handed != b["handed"]),
    ]


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------

OK = (120, 235, 120)
WAIT = (200, 200, 200)
BAD = (110, 110, 240)


def text(canvas, s, xy, scale=0.6, color=WAIT, weight=1):
    cv2.putText(canvas, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), weight + 3, cv2.LINE_AA)
    cv2.putText(canvas, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, weight, cv2.LINE_AA)


def draw_ring(canvas, center, radius, progress, active):
    """Android-style progress arc: filling means the position is being held."""
    cv2.circle(canvas, center, radius, (70, 70, 70), 3, cv2.LINE_AA)
    if progress > 0:
        cv2.ellipse(canvas, center, (radius, radius), -90, 0, 360 * progress,
                    OK if active else (0, 165, 235), 6, cv2.LINE_AA)


def draw_checklist(canvas, steps, current):
    x, y = canvas.shape[1] - 300, 40
    text(canvas, "ENROLLMENT", (x, y), 0.55, (235, 235, 235))
    for i, st in enumerate(steps):
        y += 26
        if st.done:
            mark, color = "[x]", OK
        elif st.skipped:
            mark, color = "[-]", (140, 140, 140)
        elif i == current:
            mark, color = " > ", (0, 215, 255)
        else:
            mark, color = "[ ]", (120, 120, 120)
        text(canvas, f"{mark} {st.prompt}", (x, y), 0.48, color)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def next_session_path(label):
    REC_DIR.mkdir(exist_ok=True)
    n = 1
    while (p := REC_DIR / f"{label}-{n:02d}.csv").exists():
        n += 1
    return p


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "session"
    ensure_model()
    cam = CameraThread()
    landmarker = build_landmarker()
    steps = build_steps()

    baseline = {}
    baseline_sizes = []
    current = 0
    last_stamp = 0.0
    frame_index = 0
    finished_at = None
    aborted = False

    try:
        while True:
            frame, stamp = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue
            if stamp == last_stamp:
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    aborted = True
                    break
                continue
            dt = min(stamp - last_stamp, 0.2) if last_stamp else 1 / 30
            last_stamp = stamp

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                int(frame_index * 1000 / 30),
            )
            frame_index += 1

            canvas = frame
            fh, fw = canvas.shape[:2]
            state = None
            if result.hand_landmarks:
                pts = np.array([[p.x * fw, p.y * fh] for p in result.hand_landmarks[0]])
                state = analyse(pts, result.handedness[0][0].category_name)
                draw_hand(canvas, pts)

            done_all = current >= len(steps)

            if not done_all:
                step = steps[current]
                # baseline is empty during step 1, whose test does not use it.
                satisfied = state is not None and step.test(state, baseline)

                if satisfied:
                    step.progress = min(1.0, step.progress + dt / HOLD_SECONDS)
                    step.frames += 1
                    step.samples.append(state)
                    if current == 0:
                        baseline_sizes.append(state.size)
                else:
                    step.progress = max(0.0, step.progress - dt / HOLD_SECONDS * DECAY)

                if step.progress >= 1.0:
                    step.done = True
                    if current == 0:
                        baseline["size"] = float(np.median(baseline_sizes))
                        baseline["handed"] = step.samples[-1].handed
                    current += 1

                center = (fw // 2, fh - 150)
                draw_ring(canvas, center, 52, step.progress, satisfied)
                text(canvas, step.prompt, (fw // 2 - 220, 60), 0.95,
                     OK if satisfied else (235, 235, 235), 2)
                text(canvas, step.hint, (fw // 2 - 220, 92), 0.55, (180, 180, 180))
                if state is None:
                    text(canvas, "no hand detected", (fw // 2 - 90, fh - 70), 0.6, BAD)
                elif not state.two_fingers:
                    text(canvas, "two fingers up (ring + pinky curled)",
                         (fw // 2 - 190, fh - 70), 0.6, BAD)
            else:
                if finished_at is None:
                    finished_at = time.perf_counter()
                text(canvas, "ENROLLMENT COMPLETE", (fw // 2 - 250, fh // 2), 1.2, OK, 3)
                text(canvas, "closing...", (fw // 2 - 60, fh // 2 + 40), 0.6, (200, 200, 200))
                if time.perf_counter() - finished_at > 2.0:
                    break

            if state is not None:
                text(canvas, f"angle {state.angle:+6.1f}   size {state.size:5.1f}"
                             f"   spread {state.spread:.2f}   {state.handed}",
                     (12, fh - 20), 0.5, (170, 170, 170))
            draw_checklist(canvas, steps, current)

            cv2.imshow("hand enrollment", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                aborted = True
                break
            if key == ord("s") and not done_all:
                steps[current].skipped = True
                current += 1
    finally:
        cam.release()
        landmarker.close()
        cv2.destroyAllWindows()

    write_report(label, steps, baseline, aborted)


def write_report(label, steps, baseline, aborted):
    done = [s for s in steps if s.done]
    print("\n" + "=" * 58)
    print("ENROLLMENT ABORTED" if aborted else
          ("ENROLLMENT COMPLETE" if len(done) == len(steps) else "ENROLLMENT PARTIAL"))
    print("=" * 58)

    for s in steps:
        mark = "done   " if s.done else ("skipped" if s.skipped else "MISSED ")
        if s.samples:
            ang = np.array([x.angle for x in s.samples])
            sz = np.array([x.size for x in s.samples])
            sp = np.array([x.spread for x in s.samples])
            print(f"  {mark} {s.prompt:<34} {s.frames:4d} frames   "
                  f"angle {ang.mean():+6.1f}  size {sz.mean():5.1f}  spread {sp.mean():.2f}")
        else:
            print(f"  {mark} {s.prompt:<34}    0 frames")

    if not any(s.samples for s in steps):
        print("\nno usable frames recorded")
        return

    path = next_session_path(label)
    with path.open("w", newline="") as f:
        cols = ["step", "angle", "size", "spread", "handed"] + \
               [f"{a}{i}" for i in range(21) for a in ("x", "y")]
        w = csv.writer(f)
        w.writerow(cols)
        for s in steps:
            for st in s.samples:
                w.writerow([s.key, f"{st.angle:.3f}", f"{st.size:.3f}",
                            f"{st.spread:.4f}", st.handed] +
                           [f"{v:.2f}" for p in st.pts for v in p])

    total = sum(len(s.samples) for s in steps)
    print(f"\nrecorded {total} frames -> {path}")
    if baseline:
        print(f"baseline hand size: {baseline['size']:.1f}px  "
              f"(first hand: {baseline['handed']})")

    if len(done) == len(steps):
        print("\nTracking verified: every prompted position was reached on command,")
        print("so the landmarks are following a real hand.")
    else:
        missed = [s.prompt for s in steps if not s.done and not s.skipped]
        if missed:
            print("\nNot reached: " + "; ".join(missed))
            print("If a position felt right but never filled the ring, the threshold")
            print("is wrong rather than your hand - tell me the on-screen numbers.")


if __name__ == "__main__":
    main()
