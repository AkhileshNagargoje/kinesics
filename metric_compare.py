"""Compare candidate finger extended/curled metrics on the enrolled poses.

Every enrollment frame is a true two-finger pose, so index/middle must read
extended and ring/pinky curled. The best metric is the one that separates
those two groups by the widest margin - a wide margin is what stops the
verdict flipping frame to frame.
"""
import csv
import numpy as np

MCP = dict(index=5, middle=9, ring=13, pinky=17)
PIP = dict(index=6, middle=10, ring=14, pinky=18)
DIP = dict(index=7, middle=11, ring=15, pinky=19)
TIP = dict(index=8, middle=12, ring=16, pinky=20)
EXT_F = ("index", "middle")
CURL_F = ("ring", "pinky")

rows = list(csv.DictReader(open("recordings/normal-light-01.csv")))
P = np.array([[[float(r[f"x{i}"]), float(r[f"y{i}"])] for i in range(21)] for r in rows])
size = np.linalg.norm(P[:, 9] - P[:, 0], axis=1)


def m_wrist(f):                      # current rule
    return (np.linalg.norm(P[:, TIP[f]] - P[:, 0], axis=1)
            - np.linalg.norm(P[:, PIP[f]] - P[:, 0], axis=1)) / size


def m_mcp(f):                        # tip distance from its own knuckle
    return np.linalg.norm(P[:, TIP[f]] - P[:, MCP[f]], axis=1) / size


def m_straight(f):                   # chord / arc-length: 1.0 = perfectly straight
    chord = np.linalg.norm(P[:, TIP[f]] - P[:, MCP[f]], axis=1)
    arc = (np.linalg.norm(P[:, PIP[f]] - P[:, MCP[f]], axis=1)
           + np.linalg.norm(P[:, DIP[f]] - P[:, PIP[f]], axis=1)
           + np.linalg.norm(P[:, TIP[f]] - P[:, DIP[f]], axis=1))
    return chord / arc


def m_angle(f):                      # interior angle at the PIP joint, degrees
    a = P[:, MCP[f]] - P[:, PIP[f]]
    b = P[:, TIP[f]] - P[:, PIP[f]]
    cos = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


for name, fn in [("wrist-dist (current)", m_wrist), ("tip-to-MCP", m_mcp),
                 ("straightness", m_straight), ("PIP angle", m_angle)]:
    ext = np.concatenate([fn(f) for f in EXT_F])
    curl = np.concatenate([fn(f) for f in CURL_F])
    lo, hi = (ext, curl) if ext.mean() < curl.mean() else (curl, ext)
    gap = hi.min() - lo.max()                     # worst-case separation
    d = abs(ext.mean() - curl.mean()) / np.sqrt((ext.std()**2 + curl.std()**2) / 2)
    print(f"{name:<22} extended {ext.mean():7.3f}+-{ext.std():.3f}   "
          f"curled {curl.mean():7.3f}+-{curl.std():.3f}   d={d:5.2f}   "
          f"worst-case gap {gap:+7.3f}")

print("\nper-finger worst case, best metric (straightness):")
for f in EXT_F + CURL_F:
    v = m_straight(f)
    print(f"  {f:<7} min {v.min():.3f}  mean {v.mean():.3f}  max {v.max():.3f}")
