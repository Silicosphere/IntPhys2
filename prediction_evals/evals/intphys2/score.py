"""
Score IntPhys 2 surprise losses saved by evals/intphys2/eval.py.

Merges the per-rank files (losses_<fs>fs_<ctxts>ctxt_rank*.pth), joins them
with the split's metadata.csv and reports, for every context length:
  - Relative accuracy: within each scene quadruplet, k_Impossible must be more
    surprising than its matched k_Possible (k = 1, 2), as in the pairwise
    protocol of IntPhys 2 (Fig. 4 left) and compute_metrics() in
    facebookresearch/jepa-intuitive-physics. Surprise = mean over time ("avg",
    the paper's default: "We use average surprise unless specified otherwise",
    App. D.1) or max over time ("max") of the prediction error.
  - Single-video accuracy at the best threshold (an oracle upper bound for the
    harder single-video protocol, Fig. 4 middle).
The paper reports, per model and per table column, the maximal accuracy over
its hyperparameters (window size, prediction length, context length; App. D.3);
here that is the best context length. Headline and breakdowns use avg surprise.

Usage (from prediction_evals/):
  python -m evals.intphys2.score \
      --results results/vjepa2_1_vitg_384/intphys_v2/intphys2-debug-fpc48 \
      --metadata /data/.../IntPhys2_Data/Debug/metadata.csv
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch

# Easy / Medium / Hard of the paper's Table 2, defined by environment exactly
# as in ComputeScoreIntPhys2.ipynb (the metadata's Difficulty column differs:
# it has an "Unknown" level).
PAPER_SPLITS = {
    "Easy": ["BasicLevel_0"],
    "Medium": ["SaltFlats_0", "DesertMap_0", "RaceTrack_0", "TropicalIsland_0"],
    "Hard": ["PLVDaylight_0", "Egypt_0", "RuralAustralia03_0", "ParkingGarage_0", "None"],
}
# Applied literally (the notebook's "None" does not match the metadata's
# "None_0"), this equals the metadata's Difficulty column: Easy 52, Medium 200,
# Hard 168 pairs on Main. The other 86 pairs (None_0, ShoppingMall_0,
# ConcertHall_0, BasketBallCourt_0, Arena_0, Opera_0) are "Other" and only
# count towards Overall.
ENV_TO_SPLIT = {env: split for split, envs in PAPER_SPLITS.items() for env in envs}


def load_losses(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "losses_*ctxt*.pth")))
    assert files, f"no losses_*.pth in {results_dir}"
    groups = {}
    for f in files:
        key = os.path.basename(f).split("ctxt")[0]  # losses_<fs>fs_<ctxts>
        groups.setdefault(key, []).append(f)
    assert len(groups) == 1, f"several runs in {results_dir}: {list(groups)}; pass a folder with one"
    files = next(iter(groups.values()))

    contexts, per_video = None, {}
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        contexts = list(d["context_lengths"])
        losses = d["losses"].float().numpy()  # [N, n_ctx, T], zero-padded in time
        for name, l in zip(d["names"], losses):
            # DistributedSampler pads with repeated videos: keep the first copy
            per_video.setdefault(os.path.splitext(str(name))[0], l)
    return files, contexts, per_video


def n_early(l, contexts):
    """Early-context entries at the start of each row (max_context_mode).

    With max_context_mode, the row for context C starts with C//2 - 1 losses
    of the first window predicted from contexts 2, 4, ..., C-2, followed by
    the sliding windows. Detected from the row lengths: the number of
    sliding windows must then be the same for every context.
    """
    lengths = [(row != 0).sum() for row in l]
    early = [c // 2 - 1 for c in contexts]
    windows = {n - e for n, e in zip(lengths, early)}
    if len(set(lengths)) > 1 and len(windows) == 1:
        return early
    return [0] * len(contexts)


def surprise(l, how, contexts, windows_only=False):
    """l: [n_ctx, T] zero-padded -> [n_ctx] surprise per context."""
    skip = n_early(l, contexts) if windows_only else [0] * len(contexts)
    out = []
    for row, k in zip(l, skip):
        valid = row[row != 0][k:]  # L1 errors are > 0; zeros are padding
        out.append(valid.max() if how == "max" else valid.mean())
    return np.array(out)


def best_threshold_acc(s_possible, s_impossible):
    s = np.concatenate([s_possible, s_impossible])
    y = np.concatenate([np.zeros_like(s_possible), np.ones_like(s_impossible)])
    best = 0.0
    for t in np.unique(s):
        best = max(best, ((s >= t) == y).mean())
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="folder with losses_*_rank*.pth")
    ap.add_argument("--metadata", required=True, help="metadata.csv of the evaluated split")
    ap.add_argument("--out", default=None, help="JSON output (default: <results>/scores.json)")
    args = ap.parse_args()

    files, contexts, per_video = load_losses(args.results)
    meta = pd.read_csv(args.metadata, keep_default_na=False)  # keep env "None" as a string
    meta["Split"] = meta["env"].map(ENV_TO_SPLIT).fillna("Other")
    meta["key"] = meta["file_name"].map(lambda p: os.path.splitext(os.path.basename(p))[0])
    missing = sorted(set(meta["key"]) - set(per_video))
    print(f"{len(files)} rank files, {len(per_video)} videos scored, {len(meta)} in metadata, "
          f"{len(missing)} missing, contexts {contexts}")

    # Matched pairs: (scene, k) -> k_Possible vs k_Impossible
    pairs = []
    for scene, g in meta.groupby("SceneIndex", sort=False):
        types = dict(zip(g["type"], g["key"]))
        for k in (1, 2):
            p, i = types.get(f"{k}_Possible"), types.get(f"{k}_Impossible")
            if p in per_video and i in per_video:
                pairs.append((p, i, g.iloc[0]))
    print(f"{len(pairs)} complete possible/impossible pairs\n")

    results = {"files": files, "contexts": contexts, "n_pairs": len(pairs), "per_context": {}}
    early_mode = any(n_early(v, contexts)[0] for v in per_video.values())
    s = {how: {k: surprise(v, how, contexts) for k, v in per_video.items()} for how in ("max", "avg")}
    # "win": sliding windows only, without the early-context (max_context_mode) losses
    s["max_win"] = {k: surprise(v, "max", contexts, windows_only=True) for k, v in per_video.items()}
    # Where the max surprise sits: share of videos whose max is an early-context loss
    from_early = []
    for ci in range(len(contexts)):
        hits = [int(np.argmax(v[ci][v[ci] != 0]) < n_early(v, contexts)[ci]) for v in per_video.values()]
        from_early.append(100 * float(np.mean(hits)))
    rows = []
    for ci, c in enumerate(contexts):
        r = {}
        for how in ("max", "avg", "max_win"):
            sp = np.array([s[how][p][ci] for p, _, _ in pairs])
            si = np.array([s[how][i][ci] for _, i, _ in pairs])
            r[f"relative_{how}"] = 100 * float((si > sp).mean())
            if how != "max_win":
                r[f"single_best_thresh_{how}"] = 100 * float(best_threshold_acc(sp, si))
        r["max_from_early_%"] = from_early[ci]
        results["per_context"][c] = r
        rows.append({"context": c, **r})
    table = pd.DataFrame(rows).set_index("context")
    print(f"max_context_mode losses detected: {early_mode}  (relative_max_win ignores them)")
    print(table.round(2).to_string())

    best_c = max(contexts, key=lambda c: results["per_context"][c]["relative_avg"])
    results["best_context"] = best_c
    results["best_relative_avg"] = results["per_context"][best_c]["relative_avg"]
    print(f"\nBest context {best_c}: relative accuracy (avg surprise, paper default) "
          f"{results['best_relative_avg']:.2f}%")

    # Per-pair correctness for every surprise measure and context
    recs = []
    for p, i, m in pairs:
        r = {k: m[k] for k in ("Split", "condition", "Camera") if k in m}
        for how in ("max", "avg", "max_win"):
            for cj, c in enumerate(contexts):
                r[f"{how}@{c}"] = s[how][i][cj] > s[how][p][cj]
        recs.append(r)
    correct = pd.DataFrame(recs)

    # Table 2 layout: per column, the best context (the paper reports each
    # model's best run per column over its hyperparameters).
    print("\nPaper Table 2 layout (relative accuracy %, best context per column):")
    table2 = {}
    for how in ("avg", "max", "max_win"):
        row = {}
        for col in ["Easy", "Medium", "Hard", "Overall"]:
            sub = correct if col == "Overall" else correct[correct["Split"] == col]
            accs = {c: 100 * sub[f"{how}@{c}"].mean() for c in contexts}
            c = max(accs, key=accs.get)
            row[col] = f"{accs[c]:.2f} (ctx {c}, {len(sub)} pairs)"
            table2.setdefault(how, {})[col] = {"acc": accs[c], "context": c, "pairs": len(sub)}
        print(f"  {how:8s} " + " | ".join(f"{k}: {v}" for k, v in row.items()))
    results["table2"] = table2

    # Breakdowns at the best context for avg surprise (Tables 2 and 3)
    results["breakdown"] = {}
    key = f"avg@{best_c}"
    for cols in (["Split"], ["condition"], ["Camera"], ["condition", "Camera"]):
        if all(c in correct for c in cols):
            b = correct.groupby(cols)[key].agg(["mean", "size"])
            b["mean"] *= 100
            name = " x ".join(cols)
            results["breakdown"][name] = {str(k): v for k, v in b["mean"].round(2).to_dict().items()}
            print(f"\nBy {name} (avg surprise, context {best_c}):\n"
                  f"{b.round(2).rename(columns={'mean': 'acc %', 'size': 'pairs'}).to_string()}")

    out = args.out or os.path.join(args.results, "scores.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
