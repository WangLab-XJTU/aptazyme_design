"""Evaluate the FC- scorer and export visualization data as CSV (paper work).

This script ONLY produces data for the paper figures (CSV files); plotting is
done separately.  Outputs land in paper/ (gitignored, never committed).

Exports:
  paper/evaluation/heldout_predictions.csv   seq, true_activity, pred_activity, pred_std
  paper/evaluation/roc_data.csv              fpr, tpr
  paper/evaluation/calibration_data.csv      bin_low, bin_high, mean_pred, mean_true, count
  paper/structure/structure_activity.csv     seq, activity, core_int, core_strong, entropy, cass_x_core
  paper/structure/bpp_examples.csv           group, seq_id, i, j, pair_prob  (few high + few low)
  paper/design/candidate_scores.csv          seq, pred_activity, prior_prob, conf_std, len
  paper/dataset/activity_distribution.csv    activity, count
  paper/dataset/model_spec.csv               name, value

Usage (from ribozyme/ root):
  python scripts/evaluate_scorer.py --model out/fcminus_gcn.pt --n-cand 2000
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, roc_curve
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.gnn_model import GCNconvConv1d
from scorer import config
from scorer.data import compute_bpps, load_rawdata
from scorer.graph import make_graph
from scorer.predict import predict_mc

PAPER = config.ROOT / "paper"


def core_feats(m):
    c = m[config.CORE, config.CORE]
    tri = c[np.triu_indices(60, 1)]
    core_int = float(tri.sum())
    core_strong = float((tri > 0.5).mean())
    p = tri[tri > 1e-6]
    ent = float(-np.sum(p * np.log(p)) / np.log(max(len(p), 2))) if len(p) else 0.0
    cass_x_core = float(m[6:13, 70:130].sum())
    return core_int, core_strong, ent, cass_x_core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(config.OUT_MODEL))
    ap.add_argument("--n-bpp-examples", type=int, default=3, help="high + low activity BPP examples each")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seqs, fc = load_rawdata()
    core_fixed = seqs[0][70:130]
    HI, LO = float(fc.max()), float(fc.min())
    print(f"data {len(seqs)}  activity range [{LO:.3f}, {HI:.3f}]  device={device}")

    ckpt = torch.load(args.model, map_location="cpu")
    model = GCNconvConv1d(node_dim=config.NODE_DIM, hidden_dim=config.HIDDEN,
                          gnn_layers=config.GNN_LAYERS, cnn_channels=config.CNN_CHANNELS,
                          cnn_kernel=config.CNN_KERNEL, output_dim=1).to(device)
    model.load_state_dict(ckpt["state"])

    # ---- held-out split (same seed/ratio as training) ----
    n = len(seqs)
    rng = np.random.RandomState(config.SEED)
    idx = rng.permutation(n)
    te_idx = idx[int(config.TRAIN_RATIO * n):]
    te_seqs, te_fc = [seqs[i] for i in te_idx], fc[te_idx]
    print(f"held-out: {len(te_seqs)}")

    # ---- held-out predictions -> CSV ----
    te_bpps = compute_bpps(te_seqs, cache_dir=config.CACHE_DIR)
    te_graphs = [make_graph(core_fixed, m[config.CORE, config.CORE]) for m in te_bpps]
    te_loader = DataLoader(te_graphs, batch_size=config.BATCH_SIZE, shuffle=False)
    pred, std = predict_mc(model, te_loader, n_mc=config.MC_PASSES)
    pred_c = np.clip(pred, LO, HI)
    (PAPER / "evaluation").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"seq": te_seqs, "true_activity": te_fc,
                  "pred_activity": pred_c, "pred_std": std}).to_csv(
        PAPER / "evaluation" / "heldout_predictions.csv", index=False)

    sp = spearmanr(pred_c, te_fc).correlation
    pe = pearsonr(pred_c, te_fc)[0]
    qhi, qlo = np.quantile(te_fc, 0.75), np.quantile(te_fc, 0.25)
    msk = (te_fc >= qhi) | (te_fc <= qlo)
    y_bin = (te_fc >= qhi)[msk].astype(int)
    auc = roc_auc_score(y_bin, pred_c[msk])
    fpr, tpr, _ = roc_curve(y_bin, pred_c[msk])
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(
        PAPER / "evaluation" / "roc_data.csv", index=False)
    print(f"  Spearman {sp:+.3f}  Pearson {pe:+.3f}  high/low AUROC {auc:.3f}")

    # calibration: bin predicted -> mean observed per bin
    nb = 20
    bins = np.linspace(LO, HI, nb + 1)
    b = np.digitize(pred_c, bins) - 1
    cal = [(bins[k], bins[k + 1], float(pred_c[b == k].mean()),
            float(te_fc[b == k].mean()), int((b == k).sum()))
           for k in range(nb) if (b == k).sum() > 0]
    pd.DataFrame(cal, columns=["bin_low", "bin_high", "mean_pred", "mean_true", "count"]).to_csv(
        PAPER / "evaluation" / "calibration_data.csv", index=False)

    # ---- structure-activity (all sequences) -> CSV ----
    print("computing structure features (all sequences)...")
    all_bpps = compute_bpps(seqs, cache_dir=config.CACHE_DIR)
    feats = np.array([core_feats(m) for m in all_bpps])
    (PAPER / "structure").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"seq": seqs, "activity": fc,
                  "core_int": feats[:, 0], "core_strong": feats[:, 1],
                  "entropy": feats[:, 2], "cass_x_core": feats[:, 3]}).to_csv(
        PAPER / "structure" / "structure_activity.csv", index=False)

    # BPP examples (few high + few low activity), long format for heatmaps
    order = np.argsort(-fc)
    ex_idx = list(order[:args.n_bpp_examples]) + list(order[-args.n_bpp_examples:])
    qhi = np.quantile(fc, 0.75)
    rows = []
    for rank, i in enumerate(ex_idx):
        m = all_bpps[i][config.CORE, config.CORE]
        group = "high" if fc[i] >= qhi else "low"
        for a in range(60):
            for j in range(60):
                rows.append((group, f"ex{rank}", a, j, float(m[a, j])))
    pd.DataFrame(rows, columns=["group", "seq_id", "i", "j", "pair_prob"]).to_csv(
        PAPER / "structure" / "bpp_examples.csv", index=False)

    # ---- dataset profile + model spec -> CSV ----
    (PAPER / "dataset").mkdir(parents=True, exist_ok=True)
    hist, edges = np.histogram(fc, bins=30)
    pd.DataFrame({"activity": edges[:-1], "count": hist}).to_csv(
        PAPER / "dataset" / "activity_distribution.csv", index=False)
    spec = [("node_dim", config.NODE_DIM), ("hidden", config.HIDDEN),
            ("gnn_layers", config.GNN_LAYERS), ("cnn_channels", config.CNN_CHANNELS),
            ("cnn_kernel", config.CNN_KERNEL), ("lr", config.LR),
            ("batch_size", config.BATCH_SIZE), ("epochs", config.EPOCHS),
            ("seed", config.SEED), ("n_train", int(len(seqs) * config.TRAIN_RATIO)),
            ("n_heldout", len(te_seqs)), ("activity_min", LO), ("activity_max", HI)]
    pd.DataFrame(spec, columns=["name", "value"]).to_csv(
        PAPER / "dataset" / "model_spec.csv", index=False)

    print(f"\npaper data written to {PAPER}")


if __name__ == "__main__":
    main()
