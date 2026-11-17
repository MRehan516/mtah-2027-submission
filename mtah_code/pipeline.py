"""Sticky-decoder pipeline for MTAH Phase 1 (numpy only).

Run from the repo root, after the match data is in ./data:
    python mtah_code/pipeline.py --team my_team --stay 0.9995 --scale 3.0 --min-seconds 2.5

Never pass or print real match numbers to any cloud AI. Share code and error messages only.
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from features import extract_features, to_evidence

FPS = 25
# Frames we decode as H/A ([start, end)). Graded frames are 11185..90651 and 110808..190197, BUT the
# second half only kicks off at frame 113742 (FAQ.md): frames 110808..113741 are graded and all D.
# Everything outside these spans is written as D (ungraded or dead).
DEFAULT_SPANS = [(11185, 90652), (113742, 190198)]


def viterbi_two_state(llr, stay):
    """llr[t] > 0 favours state 0 (H), < 0 favours state 1 (A). Returns 0/1 path."""
    T = len(llr)
    if T == 0:
        return np.zeros(0, dtype=np.int8)
    sw = np.log(1.0 - stay); st = np.log(stay)
    ll = np.stack([llr / 2.0, -llr / 2.0], axis=1)
    dp = np.empty((T, 2)); back = np.zeros((T, 2), dtype=np.int8)
    dp[0] = ll[0]
    for t in range(1, T):
        # stay in same state vs switch from the other
        s0, w0 = dp[t-1, 0] + st, dp[t-1, 1] + sw
        s1, w1 = dp[t-1, 1] + st, dp[t-1, 0] + sw
        if s0 >= w0: dp[t, 0] = s0 + ll[t, 0]; back[t, 0] = 0
        else:        dp[t, 0] = w0 + ll[t, 0]; back[t, 0] = 1
        if s1 >= w1: dp[t, 1] = s1 + ll[t, 1]; back[t, 1] = 1
        else:        dp[t, 1] = w1 + ll[t, 1]; back[t, 1] = 0
    path = np.empty(T, dtype=np.int8); path[-1] = int(dp[-1].argmax())
    for t in range(T - 1, 0, -1):
        path[t-1] = back[t, path[t]]
    return path


def merge_short(path, min_len):
    """Merge runs shorter than min_len frames into the previous run (repeat until stable)."""
    out = path.copy()
    for _ in range(5):
        changed = False; start = 0
        for i in range(1, len(out) + 1):
            if i == len(out) or out[i] != out[start]:
                if i - start < min_len and start > 0:
                    out[start:i] = out[start - 1]; changed = True
                start = i
        if not changed: break
    return out


def decode(e, spans, n_frames, stay, min_frames):
    labels = np.full(n_frames, "D", dtype="<U1")
    for a, b in spans:
        path = viterbi_two_state(e[a:b], stay)
        path = merge_short(path, min_frames)
        labels[a:b] = np.where(path == 0, "H", "A")
    return labels


def write_csv(path, labels):
    with open(path, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True, help="team name used for the output file")
    ap.add_argument("--stay", type=float, default=0.9995)
    ap.add_argument("--scale", type=float, default=3.0, help="metres per unit of evidence")
    ap.add_argument("--min-seconds", type=float, default=2.5)
    ap.add_argument("--cache", default=os.path.join(ROOT, "cache", "features.npz"))
    ap.add_argument("--out", default=os.path.join(ROOT, "submission"))
    a = ap.parse_args()

    from games import GAME, GAMES, require, source_frames
    from starter.baseline import team_of_each_track

    n = source_frames("tactical", GAME)
    if os.path.exists(a.cache):
        z = np.load(a.cache); dh, da = z["dh"], z["da"]
    else:
        meta, raw = require(["atd_meta", "atd_raw"], GAME, None)
        bx, by, dh, da = extract_features(raw, team_of_each_track(meta), n, GAMES[GAME]["pitch"])
        os.makedirs(os.path.dirname(a.cache), exist_ok=True)
        np.savez_compressed(a.cache, bx=bx, by=by, dh=dh, da=da)
    e = to_evidence(dh, da, a.scale)
    labels = decode(e, DEFAULT_SPANS, n, a.stay, int(a.min_seconds * FPS))
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, a.team + ".csv"); write_csv(path, labels)
    print("wrote", path)
    print("now run: python evaluation/validate_submission.py", path)
