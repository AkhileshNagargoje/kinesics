"""Diagnose why the two-finger pose drops out mid-hold.

Phase 2 showed the gate releasing and re-engaging every ~0.3s during what was
almost certainly one continuous hold. The event log cannot say why, because it
only records engaged/not. This records the reason, per frame:

  - was a hand detected at all
  - each finger's extended/curled verdict
  - the margin of each verdict, i.e. how close it was to flipping

Hold the pose as steadily as you can for the whole run and try not to change
anything. Every dropout it records is then a failure of the rule, not of you.

    python trace_pose.py            # 30 seconds
    python trace_pose.py 60         # longer
"""

import csv
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from phase1_landmark_view import CameraThread, build_landmarker, ensure_model
from enroll import PIP, TIP, analyse

OUT = Path(__file__).parent / "logs" / "pose_trace.csv"
FINGERS = ("index", "middle", "ring", "pinky")


def margins(pts):
    """Signed margin of each finger's extended test, normalized by hand size.

    Positive means 'extended', negative means 'curled'. A value near zero is a
    verdict on a knife edge - exactly what would flip frame to frame.
    """
    wrist = pts[0]
    size = max(float(np.linalg.norm(pts[9] - pts[0])), 1e-6)
    out = {}
    for f in FINGERS:
        tip = np.linalg.norm(pts[TIP[f]] - wrist)
        pip = np.linalg.norm(pts[PIP[f]] - wrist)
        out[f] = (tip - pip) / size
    return out


def draw_live(frame, st, m, armed, end):
    """Show, per finger, the live extended/curled verdict and its margin.

    The point is that a wrong verdict is visible while it is happening, rather
    than inferred from a CSV afterwards.
    """
    h, w = frame.shape[:2]
    if st is not None:
        for a, b in ((0, 5), (5, 6), (6, 8), (0, 9), (9, 10), (10, 12),
                     (0, 13), (13, 14), (14, 16), (0, 17), (17, 18), (18, 20)):
            p, q = st.pts[a].astype(int), st.pts[b].astype(int)
            cv2.line(frame, tuple(p), tuple(q), (90, 200, 120), 3, cv2.LINE_AA)
        for i in (8, 12, 16, 20):
            cv2.circle(frame, tuple(st.pts[i].astype(int)), 9, (60, 90, 255), -1, cv2.LINE_AA)

    y = 60
    for f in FINGERS:
        want_ext = f in ("index", "middle")
        if m is None:
            txt, col = f"{f:<7} --", (120, 120, 120)
        else:
            is_ext = m[f] > 0
            good = is_ext == want_ext
            txt = f"{f:<7} {'EXT ' if is_ext else 'curl'} {m[f]:+.3f}  {'ok' if good else 'WRONG'}"
            col = (110, 235, 110) if good else (80, 80, 255)
        cv2.putText(frame, txt, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, txt, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)
        y += 46

    if st is None:
        head, col = "NO HAND", (110, 110, 240)
    elif not armed:
        head, col = "waiting for pose...", (0, 190, 240)
    else:
        left = max(0.0, end - time.perf_counter())
        head, col = f"RECORDING  {left:4.1f}s", (110, 235, 110)
    cv2.putText(frame, head, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(frame, head, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 2, cv2.LINE_AA)


def main():
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    ensure_model()
    cam = CameraThread()
    landmarker = build_landmarker()
    time.sleep(1.0)

    # Recording starts only once the pose is actually visible. The previous
    # version counted down blind, so a run could record 30s of no hand at all
    # and the numbers looked like a rule failure instead of an empty room.
    print("show the two-finger pose to start; recording begins when it is seen")

    rows = []
    last, idx = 0.0, 0
    armed = False
    end = None
    while end is None or time.perf_counter() < end:
        frame, stamp = cam.read()
        if frame is None or stamp == last:
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            continue
        last = stamp

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(idx * 1000 / 30)
        )
        idx += 1
        h, w = frame.shape[:2]

        st, m = None, None
        if res.hand_landmarks:
            pts = np.array([[p.x * w, p.y * h] for p in res.hand_landmarks[0]])
            st = analyse(pts, res.handedness[0][0].category_name)
            m = margins(pts)

        if not armed:
            if st is not None and st.two_fingers:
                armed = True
                end = time.perf_counter() + duration
                print(f"  pose seen - recording {duration:.0f}s")
        elif st is None:
            rows.append(dict(t=stamp, hand=0, pose=0, size="", angle="",
                             **{f: "" for f in FINGERS}))
        else:
            rows.append(dict(t=stamp, hand=1, pose=int(st.two_fingers),
                             size=round(st.size, 1), angle=round(st.angle, 1),
                             **{f: round(m[f], 4) for f in FINGERS}))

        draw_live(frame, st, m, armed, end)
        cv2.imshow("pose trace", cv2.resize(frame, (640, 360)))
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cam.release()
    landmarker.close()
    cv2.destroyAllWindows()

    if not rows:
        print("\nnothing recorded - the pose was never seen.")
        print("If you were making it, the rule is wrong and the live view will show")
        print("which finger reads incorrectly.")
        return

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=["t", "hand", "pose", "size", "angle", *FINGERS])
        w_.writeheader()
        w_.writerows(rows)

    report(rows)
    print(f"\nsaved {OUT}")


def report(rows):
    n = len(rows)
    no_hand = sum(1 for r in rows if r["hand"] == 0)
    pose = sum(1 for r in rows if r["pose"] == 1)
    print("\n" + "=" * 58)
    print(f"frames {n}   hand lost {no_hand} ({100 * no_hand / n:.1f}%)"
          f"   pose held {pose} ({100 * pose / n:.1f}%)")
    print("=" * 58)

    # Which finger is responsible for each failure?
    blame = {f: 0 for f in FINGERS}
    for r in rows:
        if r["hand"] == 0 or r["pose"] == 1:
            continue
        for f in ("index", "middle"):
            if r[f] <= 0:
                blame[f] += 1
        for f in ("ring", "pinky"):
            if r[f] > 0:
                blame[f] += 1
    fails = n - pose - no_hand
    print(f"\nframes with a hand but no pose: {fails}")
    if fails:
        for f, c in sorted(blame.items(), key=lambda kv: -kv[1]):
            if c:
                print(f"  {f:<7} wrong on {c:4d} frames ({100 * c / fails:.0f}% of failures)")

    print("\nmargin per finger (want index/middle strongly +, ring/pinky strongly -)")
    print("  a margin hovering near 0 is the verdict that flips")
    for f in FINGERS:
        v = np.array([r[f] for r in rows if r["hand"] == 1])
        if len(v) == 0:
            continue
        near = 100 * np.mean(np.abs(v) < 0.08)
        print(f"  {f:<7} mean {v.mean():+.3f}  p5 {np.percentile(v, 5):+.3f}"
              f"  p95 {np.percentile(v, 95):+.3f}   within +-0.08 of flipping: {near:.0f}% of frames")

    # dropout run lengths
    runs, cur = [], 0
    for r in rows:
        if r["pose"] == 1:
            if cur:
                runs.append(cur)
            cur = 0
        else:
            cur += 1
    if cur:
        runs.append(cur)
    if runs:
        a = np.array(runs)
        print(f"\ndropouts: {len(a)}  median {np.median(a):.0f} frames "
              f"({np.median(a) / 30:.2f}s)  max {a.max()} frames ({a.max() / 30:.2f}s)")
    else:
        print("\nno dropouts - pose held continuously")


if __name__ == "__main__":
    main()
