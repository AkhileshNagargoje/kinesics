"""Phase 2: live two-finger pose detection with engagement state.

The pose is the clutch. This is the piece that decides when you mean to
scroll, so what matters here is not recall - enrollment already showed the
geometric rule catches the pose 100% of the time - but PRECISION: how often
it engages when you did not mean it.

So this is built to be left running while you work. It sits in a small
window, shows its state, logs every engagement, and reports a
false-engagements-per-hour figure at the end.

    python phase2_pose_detect.py             # small window, logs events
    python phase2_pose_detect.py --big       # full size, for close inspection

Keys: q quit | m mark the last engagement as a false positive | r reset counts

Press m whenever it lights up ENGAGED and you did not mean it. That gives
real ground truth instead of my guess at a threshold.
"""

import csv
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from phase1_landmark_view import CameraThread, build_landmarker, ensure_model, hand_size
from enroll import analyse
from trace_pose import margins

LOG_DIR = Path(__file__).parent / "logs"

# Asymmetric hysteresis: quick to engage so it feels responsive, slow to
# release so a single dropped frame mid-scroll does not break the gesture.
FRAMES_TO_ENGAGE = 3
FRAMES_TO_RELEASE = 6

# From enrollment: noise floor is ~1px RMS, ~0.9% of hand size. A hand smaller
# than this in frame is too far away for the landmarks to be trustworthy.
MIN_HAND_SIZE_PX = 55

# Schmitt trigger on the finger margins, not just on frame counts.
#
# The 30s trace showed the failure mode is not noise: ring and pinky slowly
# uncurl because holding them folded is tiring (ring averaged -0.255 while the
# pose held, +0.100 when it broke). A single threshold turns that fatigue into
# a dropped gesture. So engaging demands a clearly-made pose, while staying
# engaged tolerates the drift - you have already declared intent.
# Replaying the 30s trace through four maintain-rules, dropping the ring/pinky
# condition entirely once engaged won clearly: median hold 0.37s -> 1.93s and
# fragments 14 -> 5. Intent is declared by the strict engage; after that only
# the two fingers you actually scroll with need to stay up.
ENGAGE_EXT = 0.18      # index/middle clearly extended to start
ENGAGE_CURL = -0.12    # ring/pinky clearly curled to start
HOLD_EXT = 0.02        # index/middle merely still extended to continue


def pose_ok(m, strict):
    """strict=True for engaging, False for staying engaged."""
    if strict:
        return (m["index"] > ENGAGE_EXT and m["middle"] > ENGAGE_EXT
                and m["ring"] < ENGAGE_CURL and m["pinky"] < ENGAGE_CURL)
    return m["index"] > HOLD_EXT and m["middle"] > HOLD_EXT


@dataclass
class Engagement:
    start: float
    end: float = 0.0
    frames: int = 0
    angle_sum: float = 0.0
    size_sum: float = 0.0
    spurious: bool = False

    @property
    def duration(self):
        return (self.end or time.perf_counter()) - self.start

    @property
    def mean_angle(self):
        return self.angle_sum / max(self.frames, 1)

    @property
    def mean_size(self):
        return self.size_sum / max(self.frames, 1)


class PoseGate:
    """Raw per-frame rule -> debounced engaged/released state."""

    def __init__(self):
        self.engaged = False
        self.true_run = 0
        self.false_run = 0
        self.events: list[Engagement] = []

    def update(self, state, now):
        if state is None or state.size < MIN_HAND_SIZE_PX:
            raw = False
        else:
            # Threshold depends on the state we are already in.
            raw = pose_ok(margins(state.pts), strict=not self.engaged)

        if raw:
            self.true_run += 1
            self.false_run = 0
        else:
            self.false_run += 1
            self.true_run = 0

        if not self.engaged and self.true_run >= FRAMES_TO_ENGAGE:
            self.engaged = True
            self.events.append(Engagement(start=now))
        elif self.engaged and self.false_run >= FRAMES_TO_RELEASE:
            self.engaged = False
            self.events[-1].end = now

        if self.engaged and state is not None:
            e = self.events[-1]
            e.frames += 1
            e.angle_sum += state.angle
            e.size_sum += state.size

        return raw


def badge(canvas, engaged, raw, hand):
    h, w = canvas.shape[:2]
    if engaged:
        label, color = "ENGAGED", (110, 235, 110)
    elif hand:
        label, color = "pose no" if not raw else "arming", (0, 190, 240)
    else:
        label, color = "no hand", (120, 120, 130)

    cv2.rectangle(canvas, (0, 0), (w, 54), (0, 0, 0), -1)
    if engaged:
        cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), color, 6)
    cv2.putText(canvas, label, (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)


def line(canvas, s, y, color=(180, 180, 180), scale=0.5):
    cv2.putText(canvas, s, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, s, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def main():
    big = "--big" in sys.argv
    ensure_model()
    cam = CameraThread()
    landmarker = build_landmarker()
    gate = PoseGate()

    started = time.perf_counter()
    last_stamp = 0.0
    frame_index = 0
    infer_ms = deque(maxlen=60)
    flash_until = 0.0

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
            last_stamp = stamp
            now = time.perf_counter()

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t0 = time.perf_counter()
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                int(frame_index * 1000 / 30),
            )
            infer_ms.append((time.perf_counter() - t0) * 1000)
            frame_index += 1

            fh, fw = frame.shape[:2]
            state = None
            if result.hand_landmarks:
                pts = np.array([[p.x * fw, p.y * fh] for p in result.hand_landmarks[0]])
                state = analyse(pts, result.handedness[0][0].category_name)

            raw = gate.update(state, now)

            canvas = frame if big else cv2.resize(frame, (640, 360))
            if state is not None:
                # Landmarks drawn on the resized canvas need resized coords.
                scale = canvas.shape[1] / fw
                px = (state.pts * scale).astype(int)
                for a, b in ((0, 5), (5, 6), (6, 8), (0, 9), (9, 10), (10, 12),
                             (0, 13), (13, 16), (0, 17), (17, 20)):
                    cv2.line(canvas, tuple(px[a]), tuple(px[b]), (90, 200, 120), 2, cv2.LINE_AA)
                anchor = ((state.pts[5] + state.pts[9]) / 2.0 * scale).astype(int)
                cv2.circle(canvas, tuple(anchor), 8, (0, 215, 255), 2, cv2.LINE_AA)

            badge(canvas, gate.engaged, raw, state is not None)

            mins = (now - started) / 60.0
            done = [e for e in gate.events if e.end]
            spurious = sum(1 for e in gate.events if e.spurious)
            ch = canvas.shape[0]
            line(canvas, f"engagements {len(gate.events)}   marked false {spurious}", ch - 62)
            line(canvas, f"uptime {mins:5.1f} min   rate {len(gate.events) / max(mins, 1e-9) * 60:.1f}/hr",
                 ch - 44)
            if state is not None:
                line(canvas, f"angle {state.angle:+6.1f}  size {state.size:5.1f}"
                             f"  infer {np.mean(infer_ms):4.1f}ms", ch - 26)
            line(canvas, "q quit   m = that was a false trigger   r reset", ch - 8,
                 (140, 140, 140), 0.45)
            if now < flash_until:
                cv2.putText(canvas, "marked", (canvas.shape[1] - 110, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 120, 255), 2, cv2.LINE_AA)

            cv2.imshow("phase 2 - pose gate", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m") and gate.events:
                gate.events[-1].spurious = True
                flash_until = now + 1.0
            if key == ord("r"):
                gate.events.clear()
                started = now
    finally:
        cam.release()
        landmarker.close()
        cv2.destroyAllWindows()

    report(gate, time.perf_counter() - started)


def report(gate, elapsed):
    events = [e for e in gate.events if e.end or e.frames]
    for e in events:
        if not e.end:
            e.end = e.start + e.duration
    mins = elapsed / 60.0

    print("\n" + "=" * 58)
    print(f"session {mins:.1f} min   engagements {len(events)}")
    print("=" * 58)
    if not events:
        print("never engaged - if you were making the pose, the rule is too strict")
        return

    durs = np.array([e.duration for e in events])
    spurious = [e for e in events if e.spurious]
    brief = [e for e in events if e.duration < 0.35]

    print(f"duration      median {np.median(durs):.2f}s   min {durs.min():.2f}s   max {durs.max():.2f}s")
    print(f"marked false  {len(spurious)}  ({len(spurious) / len(events) * 100:.0f}% of engagements)")
    print(f"very brief    {len(brief)} under 0.35s  <- likely accidental, candidates for a minimum-dwell filter")
    print(f"rate          {len(events) / max(mins, 1e-9) * 60:.1f} engagements/hour"
          f"   false {len(spurious) / max(mins, 1e-9) * 60:.1f}/hour")

    LOG_DIR.mkdir(exist_ok=True)
    n = 1
    while (path := LOG_DIR / f"gate-{n:02d}.csv").exists():
        n += 1
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_s", "duration_s", "frames", "mean_angle", "mean_size", "spurious"])
        for e in events:
            w.writerow([f"{e.start:.3f}", f"{e.duration:.3f}", e.frames,
                        f"{e.mean_angle:.1f}", f"{e.mean_size:.1f}", int(e.spurious)])
    print(f"\nsaved {path}")

    if spurious:
        sd = np.array([e.duration for e in spurious])
        gd = np.array([e.duration for e in events if not e.spurious])
        print(f"\nfalse triggers lasted {np.median(sd):.2f}s median"
              f" vs {np.median(gd):.2f}s for real ones"
              if len(gd) else "")
        print("If false ones are consistently shorter, a minimum-dwell gate fixes it")
        print("for free - no classifier needed.")


if __name__ == "__main__":
    main()
