"""Full v2 pipeline: interpolation, soft evidence, touches, out-of-bounds dead ball, ensemble decoder.
Run from the repo root:  python mtah_code/run_v2.py --team my_team
Share only code and error messages with any AI; never match data or numbers computed from it."""
import argparse, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from improve import (extract_positions, interpolate_ball, team_distances, touch_frames,
                     add_touch_evidence, out_of_bounds_dead, ensemble_vote, long_gap_dead, still_ball_dead)
from pipeline import viterbi_two_state, merge_short, write_csv, DEFAULT_SPANS, FPS

DEFAULTS = dict(max_gap_s=2.0, k=1, scale=3.0, stay=(0.999, 0.9995, 0.9999),
                touch_bonus=3.0, touch_radius=2.5, touch_dv=3.0, min_seconds=2.5,
                use_touch=True, use_oob=True, oob_seconds=1.0,
                dead_gap_s=0.0, dead_still_s=0.0)   # 0 = rule off

def build_evidence(ball, players, team, pitch, p):
    ball_f, _ = interpolate_ball(ball, int(p["max_gap_s"] * FPS))
    dh, da = team_distances(ball_f, players, team, pitch, k=p["k"])
    e = da - dh
    llr = np.where(np.isfinite(e), e / p["scale"], 0.0)          # 0 = no evidence
    if p["use_touch"]:
        fr, tm = touch_frames(ball, players, team, pitch, radius=p["touch_radius"], min_dv=p["touch_dv"])
        llr = add_touch_evidence(llr, fr, tm, bonus=p["touch_bonus"])
    dead = out_of_bounds_dead(ball, int(p["oob_seconds"] * FPS)) if p["use_oob"] else np.zeros(len(ball), bool)
    if p["dead_gap_s"] > 0:
        dead = dead | long_gap_dead(ball, int(p["dead_gap_s"] * FPS))
    if p["dead_still_s"] > 0:
        dead = dead | still_ball_dead(ball, pitch, min_frames=int(p["dead_still_s"] * FPS))
    return llr, dead

def build_labels(ball, players, team, pitch, spans, p=DEFAULTS):
    n = len(ball); llr, dead = build_evidence(ball, players, team, pitch, p)
    labels = np.full(n, "D", dtype="<U1")
    for a, b in spans:
        paths = [viterbi_two_state(llr[a:b], s) for s in p["stay"]]
        path = merge_short(ensemble_vote(paths), int(p["min_seconds"] * FPS))
        labels[a:b] = np.where(path == 0, "H", "A")
    labels[dead] = "D"
    return labels

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--team", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "submission"))
    ap.add_argument("--no-touch", action="store_true"); ap.add_argument("--no-oob", action="store_true")
    ap.add_argument("--dead-gap-s", type=float, default=0.0, help="ball unseen this many seconds => D (0 = off)")
    ap.add_argument("--dead-still-s", type=float, default=0.0, help="ball still this many seconds => D (0 = off)")
    a = ap.parse_args()
    from games import GAME, GAMES, require, source_frames
    from starter.baseline import team_of_each_track
    meta, raw = require(["atd_meta", "atd_raw"], GAME, None)
    n = source_frames("tactical", GAME)
    cache = os.path.join(ROOT, "cache", "positions.npz")
    if os.path.exists(cache):
        z = np.load(cache); ball, players = z["ball"], z["players"]
    else:
        ball, players = extract_positions(raw, n)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, ball=ball, players=players)
    p = dict(DEFAULTS); p["use_touch"] = not a.no_touch; p["use_oob"] = not a.no_oob
    p["dead_gap_s"] = a.dead_gap_s; p["dead_still_s"] = a.dead_still_s
    labels = build_labels(ball, players, np.asarray(team_of_each_track(meta)), GAMES[GAME]["pitch"], DEFAULT_SPANS, p)
    os.makedirs(a.out, exist_ok=True); path = os.path.join(a.out, a.team + ".csv"); write_csv(path, labels)
    print("wrote", path); print("next: python evaluation/validate_submission.py", path)
