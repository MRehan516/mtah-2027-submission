"""Extra evidence builders for the MTAH pipeline (numpy only). Tested on made-up data only."""
import numpy as np

def extract_positions(raw_path, n_frames, n_tracks=46):
    """ball[n,2], players[n,tracks,2] in 0..1 units; NaN where missing. Line: frame:p;p;...:ball"""
    ball = np.full((n_frames, 2), np.nan, np.float32)
    pl = np.full((n_frames, n_tracks, 2), np.nan, np.float32)
    with open(raw_path) as fh:
        for line in fh:
            parts = line.split(":")
            if len(parts) != 3: continue
            f = int(parts[0]) - 1
            if not (0 <= f < n_frames): continue
            b = parts[2].strip().split(",")
            if not b[0].startswith("N"): ball[f] = (float(b[0]), float(b[1]))
            for s, g in enumerate(parts[1].split(";")):
                if s >= n_tracks: break
                if g.startswith("N"): continue
                p = g.split(","); pl[f, s] = (float(p[0]), float(p[1]))
    return ball, pl

def interpolate_ball(ball, max_gap):
    """Linearly fill NaN gaps of at most max_gap frames. Returns (ball_filled, was_filled)."""
    out = ball.copy(); filled = np.zeros(len(ball), bool)
    ok = np.isfinite(ball[:, 0]); idx = np.flatnonzero(ok)
    for a, b in zip(idx[:-1], idx[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            t = np.arange(1, gap + 1) / (gap + 1)
            out[a+1:b] = ball[a] + t[:, None] * (ball[b] - ball[a]); filled[a+1:b] = True
    return out, filled

def team_distances(ball, players, team_of_track, pitch, k=1, chunk=20000):
    """Mean distance (m) from the ball to the k nearest players of each team. NaN if unknown."""
    n = len(ball); L, W = pitch; scale = np.array([L, W], np.float32)
    dh = np.full(n, np.nan, np.float32); da = np.full(n, np.nan, np.float32)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        d = np.linalg.norm((players[s:e] - ball[s:e, None, :]) * scale, axis=2)   # (m, tracks)
        for team, out in ((0, dh), (1, da)):
            dt = np.sort(np.where(team_of_track[None, :] == team, d, np.inf), axis=1)[:, :k]
            dt = np.where(np.isfinite(dt).all(axis=1), dt.mean(axis=1), np.nan)
            out[s:e] = dt
    return dh, da

def touch_frames(ball, players, team_of_track, pitch, fps=25, radius=2.5, min_dv=3.0, lag=3):
    """Candidate touches: ball velocity before vs after a frame differs by more than min_dv (m/s)
    while a player is within radius metres. Returns (frames, teams). Straight pass-throughs are missed."""
    L, W = pitch; scale = np.array([L, W], np.float32)
    pos = ball * scale
    v = np.full_like(pos, np.nan); v[1:] = (pos[1:] - pos[:-1]) * fps          # m/s
    n = len(pos); fr, tm = [], []
    for f in range(lag + 1, n - lag - 1):
        before = v[f-lag:f]; after = v[f+1:f+1+lag]
        if np.isnan(before).any() or np.isnan(after).any(): continue
        if np.linalg.norm(after.mean(0) - before.mean(0)) <= min_dv: continue
        d = np.linalg.norm((players[f] - ball[f]) * scale, axis=1)
        if not np.isfinite(d).any(): continue
        j = np.nanargmin(d)
        if d[j] <= radius and (not fr or f - fr[-1] > 5):
            fr.append(f); tm.append(int(team_of_track[j]))
    return np.array(fr, int), np.array(tm, int)

def add_touch_evidence(llr, frames, teams, bonus=3.0, spread=10):
    """Add +bonus (home) / -bonus (away) around each touch frame."""
    out = llr.copy()
    for f, t in zip(frames, teams):
        out[max(0, f - spread): f + spread + 1] += bonus if t == 0 else -bonus
    return out

def out_of_bounds_dead(ball, min_frames=25, margin=0.0):
    """True where the ball has been outside 0..1 for at least min_frames in a row (possible dead ball)."""
    out = ((ball[:, 0] < -margin) | (ball[:, 0] > 1 + margin) | (ball[:, 1] < -margin) | (ball[:, 1] > 1 + margin))
    res = np.zeros(len(out), bool); s = None
    for i in range(len(out) + 1):
        if i < len(out) and out[i]:
            if s is None: s = i
        else:
            if s is not None and i - s >= min_frames: res[s:i] = True
            s = None
    return res

def ensemble_vote(paths):
    """paths: list of 0/1 arrays -> majority 0/1 (ties go to the first)."""
    s = np.sum(paths, axis=0)
    return np.where(s * 2 > len(paths), 1, np.where(s * 2 < len(paths), 0, paths[0]))


def long_gap_dead(ball, min_frames):
    """True where the ball has been unknown (NaN) for at least min_frames in a row.
    Hypothesis: long tracking loss often means the ball is out of play. Check it with sanity numbers."""
    miss = ~np.isfinite(ball[:, 0]); res = np.zeros(len(miss), bool); s = None
    for i in range(len(miss) + 1):
        if i < len(miss) and miss[i]:
            if s is None: s = i
        else:
            if s is not None and i - s >= min_frames: res[s:i] = True
            s = None
    return res

def still_ball_dead(ball, pitch, fps=25, speed_thr=0.5, min_frames=75, lag=25):
    """True where the ball (when tracked) has moved less than speed_thr m/s over a lag window for at
    least min_frames in a row. Hypothesis: a ball that sits still for seconds is a dead ball."""
    L, W = pitch; pos = ball * np.array([L, W], np.float32)
    n = len(pos); speed = np.full(n, np.nan, np.float32)
    d = np.linalg.norm(pos[lag:] - pos[:-lag], axis=1) / (lag / fps)
    speed[lag:] = d
    still = np.isfinite(speed) & (speed < speed_thr)
    res = np.zeros(n, bool); s = None
    for i in range(n + 1):
        if i < n and still[i]:
            if s is None: s = i
        else:
            if s is not None and i - s >= min_frames: res[s:i] = True
            s = None
    return res
