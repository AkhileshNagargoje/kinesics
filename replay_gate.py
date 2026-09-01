"""Replay the recorded trace through the old and new gate rules.

Same 30 seconds of real hand data, so the comparison is honest: any change in
engagement continuity comes from the rule, not from a different performance.
"""
import csv
import numpy as np
from phase2_pose_detect import (FRAMES_TO_ENGAGE, FRAMES_TO_RELEASE, pose_ok)

rows = [r for r in csv.DictReader(open("logs/pose_trace.csv"))]

def run(rule):
    engaged, tru, fls = False, 0, 0
    events, held, cur = [], 0, 0
    for r in rows:
        if r["hand"] == "0":
            ok = False
        else:
            m = {f: float(r[f]) for f in ("index", "middle", "ring", "pinky")}
            ok = rule(m, engaged)
        tru, fls = (tru + 1, 0) if ok else (0, fls + 1)
        if not engaged and tru >= FRAMES_TO_ENGAGE:
            engaged, cur = True, 0
        elif engaged and fls >= FRAMES_TO_RELEASE:
            engaged = False
            events.append(cur)
        if engaged:
            held += 1; cur += 1
    if engaged:
        events.append(cur)
    return events, held

old = lambda m, eng: pose_ok(m, strict=True)      # single threshold, as before
new = lambda m, eng: pose_ok(m, strict=not eng)   # Schmitt trigger

for name, rule in (("old (single threshold)", old), ("new (schmitt)", new)):
    ev, held = run(rule)
    a = np.array(ev) if ev else np.array([0])
    print(f"{name:<24} engagements {len(ev):3d}   engaged {100*held/len(rows):5.1f}% of frames"
          f"   median hold {np.median(a)/30:5.2f}s   longest {a.max()/30:5.2f}s")

# how often does the drift-tolerant rule save an engagement the strict one drops?
m_all = [{f: float(r[f]) for f in ("index","middle","ring","pinky")}
         for r in rows if r["hand"] == "1"]
strict_ok = sum(pose_ok(m, True) for m in m_all)
loose_ok = sum(pose_ok(m, False) for m in m_all)
print(f"\nof {len(m_all)} hand-present frames: strict passes {strict_ok} "
      f"({100*strict_ok/len(m_all):.0f}%), loose passes {loose_ok} ({100*loose_ok/len(m_all):.0f}%)")

print("\nmaintain-rule variants (engage rule unchanged and strict in all):")
def variant(name, maintain):
    def rule(m, eng):
        return pose_ok(m, strict=True) if not eng else maintain(m)
    ev, held = run(rule)
    a = np.array(ev) if ev else np.array([0])
    print(f"  {name:<38} engagements {len(ev):3d}  engaged {100*held/len(rows):5.1f}%"
          f"  median {np.median(a)/30:5.2f}s  longest {a.max()/30:5.2f}s")

variant("ring/pinky < 0.30 (current new)", lambda m: pose_ok(m, False))
variant("ring/pinky ignored, i+m extended", lambda m: m["index"] > 0.02 and m["middle"] > 0.02)
variant("at least ONE of ring/pinky curled", lambda m: m["index"] > 0.02 and m["middle"] > 0.02
        and (m["ring"] < 0.15 or m["pinky"] < 0.15))
variant("i+m extended AND clearly > ring", lambda m: m["index"] > 0.02 and m["middle"] > 0.02
        and m["ring"] < m["index"] - 0.25 and m["ring"] < m["middle"] - 0.25)
