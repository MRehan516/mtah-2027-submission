import numpy as np
def extract_features(raw_path, team_of_track, n_frames, pitch):
    """Per video frame: ball x,y (m), nearest home dist, nearest away dist (m). NaN if unknown."""
    L, W = pitch
    bx = np.full(n_frames, np.nan); by = np.full(n_frames, np.nan)
    dh = np.full(n_frames, np.nan); da = np.full(n_frames, np.nan)
    with open(raw_path) as fh:
        for line in fh:
            parts = line.split(":")
            if len(parts) != 3: continue
            ball = parts[2].strip().split(",")
            if ball[0].startswith("N"): continue
            f = int(parts[0]) - 1
            x = float(ball[0]) * L; y = float(ball[1]) * W
            bx[f] = x; by[f] = y
            best = [np.inf, np.inf]
            for slot, g in enumerate(parts[1].split(";")):
                if g.startswith("N"): continue
                p = g.split(",")
                d = np.hypot(float(p[0]) * L - x, float(p[1]) * W - y)
                t = team_of_track[slot]
                if d < best[t]: best[t] = d
            if np.isfinite(best[0]): dh[f] = best[0]
            if np.isfinite(best[1]): da[f] = best[1]
    return bx, by, dh, da

def to_evidence(dh, da, scale=3.0):
    """>0 means home nearer. 0 (no evidence) when either distance unknown."""
    e = da - dh
    return np.where(np.isfinite(e), e / scale, 0.0)

if __name__ == "__main__":
    import tempfile, os
    lines = ["1:0.5,0.5,0;0.9,0.1,0:0.52,0.5,0",     # slot0 home near ball, slot1 away far
             "2:0.5,0.5,0;0.51,0.5,0:0.7,0.7,0",     # away closer to the ball
             "3:NaN,NaN,NaN;0.9,0.1,0:NaN,NaN,NaN"]  # ball missing
    p = os.path.join(tempfile.mkdtemp(), "t.txt"); open(p, "w").write("\n".join(lines))
    bx, by, dh, da = extract_features(p, np.array([0, 1]), 3, (105.3, 68.0))
    print(np.round(dh,1), np.round(da,1)); print(to_evidence(dh, da))
