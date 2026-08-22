"""Model ablation: effect of GCN layers, Conv1d kernel size and channels.

Trains one scorer per (gnn_layers, cnn_kernel, cnn_channels) combo and records
held-out metrics.  Results -> paper/ablation/ablation_results.csv (paper data,
gitignored).

Usage (from ribozyme/ root):
  # quick local check (small subset, few epochs, reduced grid)
  python scripts/ablate_scorer.py --subset 1000 --epochs 6 \
      --layers 2 3 --kernels 3 5 --channels 32 64

  # full study (server / full data)
  python scripts/ablate_scorer.py --subset 0 --epochs 60 \
      --layers 1 2 3 4 --kernels 3 5 7 --channels 32 64
"""
import argparse
import sys
import time
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

OUT_CSV = config.ROOT / "paper" / "ablation" / "ablation_results.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=0, help="use only first N of 16384 (0 = all)")
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--kernels", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--channels", type=int, nargs="+", default=[32, 64])
    ap.add_argument("--hidden", type=int, default=config.HIDDEN)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seqs, fc = load_rawdata()
    pistol = load_pistol()
    aptamer, core_fixed = seqs[0][:70], seqs[0][70:130]
    print(f"device={device}  grid: layers={args.layers} kernels={args.kernels} "
          f"channels={args.channels}  subset={args.subset}")

    # graphs are architecture-independent -> build once, reuse across combos
    g_tr, g_te = build_datasets(seqs, fc, pistol, aptamer, core_fixed,
                                cache_dir=config.CACHE_DIR, subset=args.subset)
    print(f"train graphs {len(g_tr)}  test {len(g_te)}")

    rows = []
    for L in args.layers:
        for K in args.kernels:
            for C in args.channels:
                t0 = time.time()
                model, tmetrics, _ = train(g_tr, g_te, hidden=args.hidden, layers=L,
                                           cnn_channels=C, cnn_kernel=K,
                                           batch_size=args.batch_size,
                                           epochs=args.epochs)
                elapsed = time.time() - t0

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

                rows.append({"gnn_layers": L, "cnn_kernel": K, "cnn_channels": C,
                             "hidden": args.hidden, "heldout_spearman": round(sp, 4),
                             "heldout_auc": round(auc, 4),
                             "train_time_s": round(elapsed, 1),
                             "epochs_run": tmetrics["epochs_run"]})
                print(f"  L={L} K={K} C={C}: Spearman {sp:+.3f}  AUROC {auc:.3f}  "
                      f"({elapsed:.0f}s, {tmetrics['epochs_run']} ep)")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nsaved -> {OUT_CSV}")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
