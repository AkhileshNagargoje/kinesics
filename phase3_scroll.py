"""Phase 3: hand motion -> real scrolling.

Everything before this was measurement. This is the part where the feel is
decided, and where the project succeeds or fails.

Design, and why:

  Direct drag, not rate control. Pinch-and-drag on a phone is the mental model
  everyone already has: the page follows your hand 1:1 and stops when you stop.
  Rate control (hold hand off-centre, page scrolls steadily) needs no clutching
  but feels like a joystick, and nothing else on a desktop behaves that way.

  Scale normalization. Enrollment measured your hand between 75 and 268px - a
  3.6x range. Dividing displacement by hand size is what stops the same
  physical motion scrolling 3.6x further when you lean in.

  Pose-onset latch. When the gate engages, the anchor already has history from
  the frames where your hand was rising into view. Diffing against those emits
  one large spike. So engaging latches a fresh origin and suppresses output for
  a moment.

  One Euro filter. Landmarks jitter ~1px RMS (measured). A moving average would
  add lag you can feel; One Euro smooths hard when still and barely at all when
  moving fast, which is exactly the tradeoff scrolling wants.

  Momentum. A flick keeps scrolling and decays, and re-engaging cancels it -
  the reflex everyone has from phones.

Starts DISARMED - it will not touch your scroll until you press 'a'.

Keys: a arm/disarm | i invert direction | [ ] gain | m momentum on/off | q quit
"""

import ctypes
import pathlib
import sys
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from phase1_landmark_view import CameraThread, build_landmarker, ensure_model
from enroll import analyse
from phase2_pose_detect import MIN_HAND_SIZE_PX, FRAMES_TO_ENGAGE, FRAMES_TO_RELEASE, pose_ok

# --------------------------------------------------------------------------
# scroll injection
# --------------------------------------------------------------------------

WHEEL_DELTA = 120                      # one notch
MOUSEEVENTF_WHEEL = 0x0800
INPUT_MOUSE = 0

ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong), ("dwExtraInfo", ULONG_PTR)]


class _U(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", ctypes.c_ulong), ("u", _U)]


class Wheel:
    """Sends wheel events, accumulating the fraction that does not fit.

    Sub-notch deltas are sent as-is rather than rounded to 120: modern apps
    honour them and that is what makes the scroll continuous instead of
    stepping. The accumulator keeps small movements from being lost.
    """

    def __init__(self):
        self.residual = 0.0
        self.sent_units = 0.0

    def scroll(self, units):
        self.residual += units
        whole = int(self.residual)
        if whole == 0:
            return
        self.residual -= whole
        self.sent_units += whole
        inp = INPUT(type=INPUT_MOUSE,
                    mi=MOUSEINPUT(0, 0, ctypes.c_ulong(whole & 0xFFFFFFFF),
                                  MOUSEEVENTF_WHEEL, 0, 0))
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


# --------------------------------------------------------------------------
# filtering
# --------------------------------------------------------------------------

class OneEuro:
    """Low lag when moving fast, heavy smoothing when nearly still."""

    def __init__(self, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0

    def __call__(self, x, dt):
        if self.x_prev is None:
            self.x_prev = x
            return x
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx_prev = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(self.dx_prev)
        a = self._alpha(cutoff, dt)
        self.x_prev = a * x + (1 - a) * self.x_prev
        return self.x_prev


# --------------------------------------------------------------------------
# tuning
# --------------------------------------------------------------------------

GAIN = 350.0            # wheel units per 1.0 of normalized hand displacement
#   900 was measured at ~19 notches (two screenfuls) per gesture - too fast.
ONSET_SUPPRESS = 0.10   # seconds of silence after engaging, to kill the jump
DEADZONE = 0.0015       # normalized; measured jitter is ~0.009 of hand size
FLICK_MIN = 0.9         # normalized units/sec to trigger momentum
MOMENTUM_DECAY = 0.93   # per frame; ~1.6s of coast from a typical flick
# In normalized units per frame. A typical flick starts around 0.05, so the
# floor has to be well below that or the coast dies in a few frames. At the
# default gain 0.0015 is about one wheel unit - genuinely imperceptible.
MOMENTUM_FLOOR = 0.0015


class Scroller:
    def __init__(self):
        self.wheel = Wheel()
        self.fx = OneEuro()
        self.fy = OneEuro()
        self.engaged = False
        self.true_run = self.false_run = 0
        self.origin = None
        self.engaged_at = 0.0
        self.last_y = None
        self.vel = deque(maxlen=5)
        self.momentum = 0.0
        self.armed = False
        self.invert = False
        self.gain = GAIN
        self.use_momentum = True
        self.last_emit = 0.0
        self.total_scrolled = 0.0
        self.gestures = 0
        self.trace = []

    def update(self, state, dt, now):
        raw = (state is not None and state.size >= MIN_HAND_SIZE_PX
               and pose_ok(_margins(state.pts), strict=not self.engaged))

        if raw:
            self.true_run += 1
            self.false_run = 0
        else:
            self.false_run += 1
            self.true_run = 0

        if not self.engaged and self.true_run >= FRAMES_TO_ENGAGE:
            self._engage(state, now)
        elif self.engaged and self.false_run >= FRAMES_TO_RELEASE:
            self._release()

        emitted = 0.0
        if self.engaged and state is not None:
            emitted = self._drag(state, dt, now)
        elif self.momentum:
            emitted = self._coast()

        self.last_emit = emitted
        return raw, emitted

    def _engage(self, state, now):
        self.engaged = True
        self.gestures += 1
        self.momentum = 0.0            # catching a coasting page stops it
        self.fx.reset()
        self.fy.reset()
        anchor = _anchor(state.pts) / state.size
        self.origin = anchor
        self.last_y = anchor[1]
        self.engaged_at = now
        self.vel.clear()

    def _release(self):
        self.engaged = False
        if self.use_momentum and len(self.vel) >= 3:
            v = float(np.mean(self.vel))
            if abs(v) > FLICK_MIN:
                self.momentum = v / 30.0     # per-frame step
        self.origin = None
        self.last_y = None

    def _drag(self, state, dt, now):
        anchor = _anchor(state.pts) / state.size
        y = self.fy(float(anchor[1]), dt)

        if now - self.engaged_at < ONSET_SUPPRESS:
            self.last_y = y                  # track but stay silent
            return 0.0

        dy = y - self.last_y
        self.last_y = y
        self.vel.append(dy / max(dt, 1e-3))
        if abs(dy) < DEADZONE:
            return 0.0
        return self._emit(dy)

    def _coast(self):
        self.momentum *= MOMENTUM_DECAY
        if abs(self.momentum) < MOMENTUM_FLOOR:
            self.momentum = 0.0
            return 0.0
        return self._emit(self.momentum)

    def _emit(self, dy_norm):
        # Image y grows downward, so hand moving up gives dy < 0. A negative
        # wheel delta scrolls the content down - the same direction a phone
        # advances when you swipe up.
        units = dy_norm * self.gain * (-1 if self.invert else 1)
        if self.armed:
            self.wheel.scroll(units)
        self.total_scrolled += abs(units)
        return units


def _anchor(pts):
    return (pts[5] + pts[9]) / 2.0


def _margins(pts):
    from trace_pose import margins
    return margins(pts)


# --------------------------------------------------------------------------
# display
# --------------------------------------------------------------------------

def hud(canvas, sc, state, fps):
    h, w = canvas.shape[:2]
    if not sc.armed:
        label, color = "DISARMED  (press a)", (110, 110, 240)
    elif sc.engaged:
        label, color = "SCROLLING", (110, 235, 110)
    elif sc.momentum:
        label, color = "coasting", (0, 200, 250)
    else:
        label, color = "armed - show two fingers", (0, 190, 240)

    cv2.rectangle(canvas, (0, 0), (w, 46), (0, 0, 0), -1)
    cv2.putText(canvas, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    if sc.engaged and sc.armed:
        cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), color, 5)

    # scroll rate bar: length and direction of the last emitted delta
    cx, cy = w - 40, h // 2
    cv2.line(canvas, (cx, cy - 70), (cx, cy + 70), (70, 70, 70), 2)
    mag = int(np.clip(sc.last_emit * 0.6, -70, 70))
    if mag:
        cv2.line(canvas, (cx, cy), (cx, cy + mag), (110, 235, 110), 7, cv2.LINE_AA)

    for i, s in enumerate([
        f"gain {sc.gain:.0f}   {'inverted' if sc.invert else 'natural'}"
        f"   momentum {'on' if sc.use_momentum else 'off'}",
        f"gestures {sc.gestures}   emitted {sc.total_scrolled:.0f} units   {fps:.0f} fps",
        "a arm   i invert   [ ] gain   m momentum   q quit",
    ]):
        y = h - 52 + i * 18
        cv2.putText(canvas, s, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, s, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (185, 185, 185), 1, cv2.LINE_AA)


def main():
    ensure_model()
    cam = CameraThread()
    landmarker = build_landmarker()
    sc = Scroller()
    big = "--big" in sys.argv

    last_stamp = 0.0
    frame_index = 0
    frame_h = 720
    fps_hist = deque(maxlen=30)

    print("phase 3 running - DISARMED. Focus the window and press 'a' to arm.")
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
            frame_h = h
            state = None
            if res.hand_landmarks:
                pts = np.array([[p.x * w, p.y * h] for p in res.hand_landmarks[0]])
                state = analyse(pts, res.handedness[0][0].category_name)

            raw_ok, emitted = sc.update(state, dt, now)
            sc.trace.append(dict(
                y_px=float(_anchor(state.pts)[1]) if state is not None else np.nan,
                size=state.size if state is not None else np.nan,
                emit=emitted, engaged=int(sc.engaged),
                hand=int(state is not None)))

            canvas = frame if big else cv2.resize(frame, (640, 360))
            if state is not None:
                s = canvas.shape[1] / w
                for a, b in ((0, 5), (5, 6), (6, 8), (0, 9), (9, 10), (10, 12), (0, 17)):
                    cv2.line(canvas, tuple((state.pts[a] * s).astype(int)),
                             tuple((state.pts[b] * s).astype(int)), (90, 200, 120), 2, cv2.LINE_AA)
                cv2.circle(canvas, tuple((_anchor(state.pts) * s).astype(int)), 8,
                           (0, 215, 255), 2, cv2.LINE_AA)

            hud(canvas, sc, state, np.mean(fps_hist))
            cv2.imshow("phase 3 - gesture scroll", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("a"):
                sc.armed = not sc.armed
                print("ARMED" if sc.armed else "disarmed")
            elif key == ord("i"):
                sc.invert = not sc.invert
            elif key == ord("m"):
                sc.use_momentum = not sc.use_momentum
            elif key == ord("]"):
                sc.gain *= 1.25
            elif key == ord("["):
                sc.gain /= 1.25
    finally:
        cam.release()
        landmarker.close()
        cv2.destroyAllWindows()

    print(f"\n{sc.gestures} scroll gestures, {sc.total_scrolled:.0f} wheel units emitted")
    print(f"final gain {sc.gain:.0f}, {'inverted' if sc.invert else 'natural'} direction")
    save_trace(sc.trace)
    direction_report(sc.trace, frame_h)


def save_trace(trace):
    if not trace:
        return
    import csv
    d = pathlib.Path(__file__).parent / "logs"
    d.mkdir(exist_ok=True)
    n = 1
    while (f := d / f"scroll-{n:02d}.csv").exists():
        n += 1
    with f.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(trace[0]))
        w.writeheader(); w.writerows(trace)
    print(f"trace saved {f}")


def direction_report(trace, frame_h):
    """Why one scroll direction works better than the other.

    Compares upward and downward hand strokes on the things that could break
    asymmetrically: where in the frame the hand is, how big it reads, and
    whether the gate survives the stroke.
    """
    if not trace:
        return
    a = np.array([[r["y_px"], r["size"], r["emit"], r["engaged"], r["hand"]] for r in trace],
                 dtype=float)
    y_px, size, emit, eng, hand = a.T

    print("\n" + "=" * 58)
    print("direction breakdown")
    print("=" * 58)

    moving = np.abs(emit) > 0
    up = moving & (emit < 0)        # hand moving up in frame
    down = moving & (emit > 0)
    print(f"frames emitting: up {up.sum():4d}   down {down.sum():4d}")
    if up.sum() and down.sum():
        print(f"hand height in frame   up  median {np.nanmedian(y_px[up]):5.0f}px"
              f"   down median {np.nanmedian(y_px[down]):5.0f}px   (frame is {frame_h}px tall)")
        print(f"hand size              up  median {np.nanmedian(size[up]):5.0f}px"
              f"   down median {np.nanmedian(size[down]):5.0f}px")
        print(f"emitted per frame      up  median {np.median(np.abs(emit[up])):5.1f}"
              f"   down median {np.median(np.abs(emit[down])):5.1f} units")

    # Loss while a gesture is live is the number that matters. Loss while idle
    # is just an empty room and inflates the overall figure.
    during = eng == 1
    if during.any() and (~during).any():
        print(f"\nhand lost during gestures: {100 * (hand[during] == 0).mean():.1f}%"
              f"   while idle: {100 * (hand[~during] == 0).mean():.1f}%")

    lost = hand == 0
    if lost.any():
        seen = y_px[hand == 1]
        print(f"\nhand lost in {100 * lost.mean():.1f}% of frames")
        print(f"hand y when present: p5 {np.percentile(seen, 5):.0f}  "
              f"median {np.median(seen):.0f}  p95 {np.percentile(seen, 95):.0f}px")
        low = (seen > frame_h * 0.75).mean()
        print(f"{100 * low:.0f}% of the time the hand sits in the bottom quarter of frame,")
        print("which is where landmarks degrade and the gate drops mid-stroke.")

    # gate releases split by which way the hand was travelling
    rel_up = rel_down = 0
    for i in range(1, len(eng)):
        if eng[i - 1] and not eng[i]:
            recent = emit[max(0, i - 6):i]
            if len(recent) and np.abs(recent).sum():
                (rel_up := rel_up + 1) if recent.sum() < 0 else (rel_down := rel_down + 1)
    print(f"\ngate releases while moving up {rel_up}   while moving down {rel_down}")
    if rel_down > rel_up * 1.5:
        print("-> the gate is dropping mainly on downward strokes; that is the asymmetry")


if __name__ == "__main__":
    main()
