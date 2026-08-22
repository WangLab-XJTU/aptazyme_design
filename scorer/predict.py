"""Inference: score sequences -> prior probability + confidence -> design batch."""
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from .config import MC_PASSES, N_JOBS


def predict_mc(model, loader, n_mc=MC_PASSES):
    """MC-dropout: mean + std of pred activity over n_mc forward passes.

    Prints progress once per batch.
    """
    device = next(model.parameters()).device
    model.train()
    n_batches = len(loader)
    means, stds = [], []
    with torch.no_grad():
        for bi, batch in enumerate(loader, 1):
            batch = batch.to(device)
            outs = torch.stack([model(batch) for _ in range(n_mc)]).cpu()  # (mc, B)
            means.append(outs.mean(0).numpy())
            stds.append(outs.std(0).numpy())
            print(f"  [predict] batch {bi}/{n_batches} ({batch.num_graphs} seqs)",
                  flush=True)
    return np.concatenate(means), np.concatenate(stds)


def prior_prob(pred, sorted_fc):
    """Activity prior = fraction of measured FC- at or below this prediction."""
    return np.searchsorted(sorted_fc, pred) / len(sorted_fc)


def score_sequences(model, full_seqs, slice_idx=None, cache_dir=None,
                    lo=None, hi=None, sorted_fc=None, batch_size=256,
                    n_mc=MC_PASSES, j=N_JOBS):
    """Score the ribozyme domain of FULL-LENGTH sequences -> scoring table.

    BPP is computed on the full sequence (so the score reflects the ribozyme's
    activity in its full-length context); the domain submatrix is sliced out
    and scored.

    slice_idx:
      None                     -> score each whole sequence as the ribozyme
      (start, stop)            -> same slice for all sequences
      list of (start, stop)/None -> per-sequence slice

    cache_dir=None (default) keeps all BPP matrices in memory (no disk cache);
    pass a cache_dir to persist/reuse BPP on disk instead.

    Returns DataFrame[seq, pred_activity, conf_std, prior_prob?, len].
    lo/hi clamp predictions to the measured activity range; sorted_fc enables
    the activity prior probability column.
    """
    from .data import compute_bpps
    from .graph import make_graph
    bpps = compute_bpps(full_seqs, cache_dir=cache_dir, j=j)
    if isinstance(slice_idx, tuple):
        slices = [slice_idx] * len(full_seqs)
    elif slice_idx is None:
        slices = [None] * len(full_seqs)
    else:
        slices = list(slice_idx)
    graphs = []
    for s, m, sl in zip(full_seqs, bpps, slices):
        if sl is None:
            graphs.append(make_graph(s, m))
        else:
            start, stop = sl
            graphs.append(make_graph(s[start:stop], m[start:stop, start:stop]))
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    pm, ps = predict_mc(model, loader, n_mc=n_mc)
    if lo is not None and hi is not None:
        pm = np.clip(pm, lo, hi)
    df = pd.DataFrame({"seq": full_seqs, "pred_activity": pm, "conf_std": ps,
                       "len": [len(s) for s in full_seqs]})
    if sorted_fc is not None:
        df["prior_prob"] = prior_prob(pm, sorted_fc)
    return df
