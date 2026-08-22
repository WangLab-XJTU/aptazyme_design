"""Train the FC- GCN scorer (16384 real FC- + 676 pistol cores).

Usage (from ribozyme/ root):
    python scripts/train_scorer.py --subset 0 --epochs 60 --batch-size 256
    python scripts/train_scorer.py --subset 4000 --epochs 25   # quick CPU check
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scorer import config
from scorer.data import load_pistol, load_rawdata
from scorer.train import build_datasets, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=0, help="use only first N of 16384 (0 = all)")
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--hidden", type=int, default=config.HIDDEN)
    ap.add_argument("--layers", type=int, default=config.GNN_LAYERS)
    args = ap.parse_args()

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seqs, fc = load_rawdata()
    pistol = load_pistol()
    aptamer, core_fixed = seqs[0][:70], seqs[0][70:130]
    print(f"rawdata {len(seqs)}  pistol {len(pistol)}  HIGH={float(fc.max()):.3f}  device={device}")

    g_tr, g_te = build_datasets(seqs, fc, pistol, aptamer, core_fixed,
                                cache_dir=config.CACHE_DIR, subset=args.subset)
    model, tmetrics, curve = train(g_tr, g_te, hidden=args.hidden, layers=args.layers,
                                   batch_size=args.batch_size, epochs=args.epochs)
    torch.save({"state": model.state_dict(), "config": vars(args)}, config.OUT_MODEL)

    # learning curve -> paper/learning/training_curve.csv (paper data, gitignored)
    learning_dir = config.ROOT / "paper" / "learning"
    learning_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(curve, columns=["epoch", "train_loss", "heldout_spearman"]).to_csv(
        learning_dir / "training_curve.csv", index=False)

    # ---- held-out validation (real FC- labels) ----
    te_loader = DataLoader(g_te, batch_size=args.batch_size, shuffle=False)
    model.eval()
    with torch.no_grad():
        pes, yes = [], []
        for b in te_loader:
            b = b.to(device)
            pes.append(model(b).cpu())
            yes.append(b.y.squeeze(-1).cpu())
        pe = torch.cat(pes).numpy()
        ye = torch.cat(yes).numpy()
    sp = spearmanr(pe, ye).correlation
    qhi, qlo = np.quantile(ye, 0.75), np.quantile(ye, 0.25)
    msk = (ye >= qhi) | (ye <= qlo)
    auc = roc_auc_score((ye >= qhi)[msk].astype(int), pe[msk])
    print(f"\nheld-out (n={len(te_loader.dataset)}): Spearman {sp:+.3f}  "
          f"high/low AUROC {auc:.3f}")

    report = {"train": tmetrics, "heldout": {"n": len(g_te),
              "spearman": float(sp), "auc_hi_lo": float(auc)},
              "n_train_graphs": len(g_tr), "n_test": len(g_te),
              "pistol_label": float(fc.max()),
              "config": vars(args)}
    with open(config.OUT_DIR / "fcminus_gcn_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved -> {config.OUT_MODEL} + report.json")


if __name__ == "__main__":
    main()
