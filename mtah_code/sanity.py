"""Checks you can run on your own labels (no ground truth needed). Prints only summary numbers.
Run:  python mtah_code/sanity.py submission/my_team.csv
Do NOT paste its output, or any real numbers, into any cloud AI."""
import sys
import numpy as np

SPANS = [(11185, 90652), (110808, 190198)]
FPS = 25

def summarise(labels):
    rows = {}
    for a, b in SPANS:
        seg = labels[a:b]
        for name in "HAD":
            rows[name] = rows.get(name, 0) + int((seg == name).sum())
    total = sum(rows.values())
    runs = []
    for a, b in SPANS:
        seg = labels[a:b]; s = 0
        for i in range(1, len(seg) + 1):
            if i == len(seg) or seg[i] != seg[s]:
                if seg[s] in "HA": runs.append(i - s)
                s = i
    runs = np.array(runs) / FPS
    return {"share": {k: round(v / total, 3) for k, v in rows.items()},
            "possessions": len(runs), "mean_s": round(float(runs.mean()), 1),
            "median_s": round(float(np.median(runs)), 1),
            "under_2s": int((runs < 2).sum()), "over_60s": int((runs > 60).sum())}

if __name__ == "__main__":
    lab = np.loadtxt(sys.argv[1], delimiter=",", skiprows=1, dtype=str)[:, 1]
    print(summarise(lab))
