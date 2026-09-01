import csv, sys
from collections import defaultdict
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "recordings/normal-light-01.csv"
rows = list(csv.DictReader(open(path)))
by = defaultdict(list)
for r in rows:
    by[r["step"]].append(r)

def pts_of(r):
    return np.array([[float(r[f"x{i}"]), float(r[f"y{i}"])] for i in range(21)])

print(f"{len(rows)} frames, {len(by)} steps\n")

# --- 1. anchor jitter while holding still -> deadzone -----------------------
print("anchor jitter per step (residual vs 9-frame baseline, px):")
all_norm = []
for step, rs in by.items():
    if len(rs) < 25: continue
    P = np.array([pts_of(r) for r in rs])
    anchor = (P[:,5] + P[:,9]) / 2.0
    size = np.array([float(r["size"]) for r in rs])
    k = 9; ker = np.ones(k)/k
    bx = np.convolve(anchor[:,0], ker, "same"); by_ = np.convolve(anchor[:,1], ker, "same")
    resid = np.hypot(anchor[:,0]-bx, anchor[:,1]-by_)[k:-k]
    norm = resid / size[k:-k]
    all_norm.append(norm)
    print(f"  {step:<12} rms {resid.mean():5.2f}px  p95 {np.percentile(resid,95):5.2f}px"
          f"   normalized {norm.mean()*100:5.2f}% of hand size")
n = np.concatenate(all_norm)
print(f"\n  pooled normalized jitter: mean {n.mean()*100:.2f}%  p95 {np.percentile(n,95)*100:.2f}%"
      f"  p99 {np.percentile(n,99)*100:.2f}% of hand size")

# --- 2. ranges -------------------------------------------------------------
size = np.array([float(r["size"]) for r in rows])
ang  = np.array([float(r["angle"]) for r in rows])
spr  = np.array([float(r["spread"]) for r in rows])
print(f"\nsize   {size.min():6.1f} .. {size.max():6.1f} px   ({size.max()/size.min():.1f}x range)")
print(f"angle  {ang.min():+6.1f} .. {ang.max():+6.1f} deg")
up = np.array([float(r["angle"]) for r in by["upright"]])
print(f"natural neutral angle: {np.median(up):+.1f} deg  (not 0 - this is your resting tilt)")

# --- 3. spread separation --------------------------------------------------
tog = np.array([float(r["spread"]) for r in by["together"]])
spd = np.array([float(r["spread"]) for r in by["spread"]])
nat = np.array([float(r["spread"]) for r in by["upright"]])
print(f"\nspread  together {tog.mean():.2f}+-{tog.std():.2f}   natural {nat.mean():.2f}"
      f"   spread {spd.mean():.2f}+-{spd.std():.2f}")
print(f"        separation: {(spd.mean()-tog.mean())/np.hypot(spd.std(),tog.std()):.1f} sigma")

# --- 4. does the two-finger rule hold across ALL recorded variation? --------
TIP={"i":8,"m":12,"r":16,"p":20}; PIP={"i":6,"m":10,"r":14,"p":18}
def ext(P,k):
    return np.linalg.norm(P[TIP[k]]-P[0]) > np.linalg.norm(P[PIP[k]]-P[0])
ok = 0
fails = defaultdict(int)
for r in rows:
    P = pts_of(r)
    if ext(P,"i") and ext(P,"m") and not ext(P,"r") and not ext(P,"p"):
        ok += 1
    else:
        fails[r["step"]] += 1
print(f"\ngeometric two-finger rule fires on {100*ok/len(rows):.1f}% of enrolled frames")
if fails:
    print("  misses:", dict(fails))
