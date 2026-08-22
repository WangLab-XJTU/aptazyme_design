"""Score the ribozyme domain of full-length sequences -> scoring table.

Input (CSV, recommended -- supports a per-sequence slice):
    name,seq,slice_start,slice_stop
    seq1,<full-length sequence>,70,130
    seq2,<full-length sequence>,55,115
  Columns: `seq` required; `slice_start`/`slice_stop` (or `start`/`stop`)
  optional per sequence; `name` optional.
  SLICES ARE 0-BASED PYTHON-STYLE: slice_start=70, slice_stop=130 means the
  domain is sequence[70:130] (i.e. 1-based positions 71-130).

  FASTA is also accepted (`--in x.fasta`); then --slice applies to all
  sequences (omit --slice to score the whole sequence).

BPP is computed on the full sequence so the score reflects the ribozyme's
activity in its full-length context; the domain submatrix at the slice is
scored by the trained model.

Usage:
    python scripts/score_design.py --model out/fcminus_gcn.pt --in design.csv
    python scripts/score_design.py --model out/fcminus_gcn.pt --in full.fasta --slice 70 130

Output (--out, default out/scoring_table.csv):
    name, seq, slice, pred_activity, conf_std, prior_prob, len
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.gnn_model import GCNconvConv1d
from scorer import config
from scorer.data import load_csv_records, load_fasta, load_rawdata
from scorer.predict import score_sequences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(config.OUT_MODEL), help="trained model .pt")
    ap.add_argument("--in", dest="inp", required=True,
                    help="input CSV (with slice columns) or FASTA")
    ap.add_argument("--slice", type=int, nargs=2, default=None, metavar=("START", "STOP"),
                    help="global slice [START:STOP] (0-based) applied when the input "
                         "does not carry a per-sequence slice; omit to score whole seq")
    ap.add_argument("--limit", type=int, default=0, help="score only first N records")
    ap.add_argument("--out", default=str(config.OUT_DIR / "scoring_table.csv"))
    args = ap.parse_args()

    seqs, fc = load_rawdata()
    LO, HI = float(fc.min()), float(fc.max())

    # ---- load input (CSV preferred; FASTA fallback) ----
    if args.inp.lower().endswith((".csv", ".tsv", ".txt")):
        records = load_csv_records(args.inp, limit=args.limit)
        names = [r[0] for r in records]
        full = [r[1] for r in records]
        per_seq = [r[2] for r in records]
    else:
        full = load_fasta(args.inp, limit=args.limit)
        names = [f"seq{i}" for i in range(len(full))]
        per_seq = [None] * len(full)
    print(f"records: {len(full)}")

    global_sl = tuple(args.slice) if args.slice else None
    slices = [sl if sl is not None else global_sl for sl in per_seq]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model, map_location=device)
    model = GCNconvConv1d(node_dim=config.NODE_DIM, hidden_dim=config.HIDDEN,
                          gnn_layers=config.GNN_LAYERS, cnn_channels=config.CNN_CHANNELS,
                          cnn_kernel=config.CNN_KERNEL, output_dim=1).to(device)
    model.load_state_dict(ckpt["state"])
    print(f"device: {device}")

    df = score_sequences(model, full, slices, lo=LO, hi=HI,
                         sorted_fc=np.sort(fc), batch_size=config.BATCH_SIZE)
    df.insert(0, "name", names)
    df["slice"] = [f"{sl[0]}:{sl[1]}" if sl else "" for sl in slices]
    df.to_csv(args.out, index=False)
    print(f"\nscoring table -> {args.out}  (n={len(df)})")
    print(df[["pred_activity", "conf_std", "prior_prob"]].describe().round(3))
    print(df[["name", "slice", "pred_activity", "conf_std", "prior_prob"]].head().round(3).to_string(index=False))


if __name__ == "__main__":
    main()
