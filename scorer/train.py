"""Training loop for the FC- GCN scorer.

Builds the graph datasets (16384 constructs + 676 real pistol cores, the latter
labeled with max FC- as a prior), trains GCNconvConv1d -> FC- with MSE, early
stops on held-out Spearman, returns the best model.
"""
import numpy as np
import torch
from scipy.stats import spearmanr
from torch_geometric.loader import DataLoader

from model.gnn_model import GCNconvConv1d

from .config import (BATCH_SIZE, CNN_CHANNELS, CNN_KERNEL, CORE, EPOCHS,
                     GNN_LAYERS, HIDDEN, LR, N_JOBS, NODE_DIM, PATIENCE, SEED,
                     TRAIN_RATIO)


def build_datasets(raw_seqs, fc, pistol_seqs, aptamer, core_fixed, cache_dir,
                   subset=0, ratio=TRAIN_RATIO, seed=SEED, j=N_JOBS):
    """Returns (g_tr, g_te) PyG graphs with .y labels; pistol cores go to train."""
    from .data import compute_bpps
    from .graph import candidate_graph, make_graph

    n = len(raw_seqs)
    if subset and subset < n:
        n = subset
    bpps = compute_bpps(raw_seqs[:n], cache_dir=cache_dir, j=j)
    graphs = [make_graph(core_fixed, m[CORE, CORE]) for m in bpps]
    for g, y in zip(graphs, fc[:n]):
        g.y = torch.FloatTensor([y])

    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_tr = int(ratio * n)
    g_tr = [graphs[i] for i in idx[:n_tr]]
    g_te = [graphs[i] for i in idx[n_tr:]]

    if pistol_seqs:
        HIGH = float(fc.max())
        pg = [candidate_graph(s, aptamer, cache_dir=cache_dir) for s in pistol_seqs]
        for g in pg:
            g.y = torch.FloatTensor([HIGH])
        g_tr = g_tr + pg
    return g_tr, g_te


def train(g_tr, g_te, hidden=HIDDEN, layers=GNN_LAYERS, cnn_channels=CNN_CHANNELS,
          cnn_kernel=CNN_KERNEL, lr=LR, batch_size=BATCH_SIZE, patience=PATIENCE,
          epochs=EPOCHS, seed=SEED):
    """Train GCNconvConv1d -> FC-; returns (best_model, metrics dict, curve list)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GCNconvConv1d(node_dim=NODE_DIM, hidden_dim=hidden, gnn_layers=layers,
                          cnn_channels=cnn_channels, cnn_kernel=cnn_kernel,
                          output_dim=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.MSELoss()
    tr_loader = DataLoader(g_tr, batch_size=batch_size, shuffle=True)
    te_loader = DataLoader(g_te, batch_size=batch_size, shuffle=False)

    best_sp, best_state, no_impr = -9, None, 0
    curve = []          # (epoch, train_loss, heldout_spearman) for learning curves
    for ep in range(1, epochs + 1):
        model.train()
        loss_acc, n_batches = 0.0, 0
        for batch in tr_loader:
            batch = batch.to(device)
            opt.zero_grad()
            loss = lossf(model(batch), batch.y.squeeze(-1))
            loss.backward()
            opt.step()
            loss_acc += loss.item()
            n_batches += 1
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
        curve.append((ep, loss_acc / max(n_batches, 1), float(sp)))
        if sp > best_sp:
            best_sp = sp
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_impr = 0
        else:
            no_impr += 1
        if ep % 10 == 0 or ep == 1:
            print(f"  ep {ep:3d} held-out Spearman {sp:+.3f}")
        if no_impr >= patience:
            print(f"  early stop @ ep {ep}")
            break

    model.load_state_dict(best_state)
    return model, {"spearman": float(best_sp), "device": str(device),
                   "epochs_run": ep, "best_epoch": ep - no_impr}, curve
