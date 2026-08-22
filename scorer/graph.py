"""Build PyG graphs: base one-hot + BPP submatrix -> nodes/edges.

Nodes = ribozyme-domain nucleotides.  Node features = [base one-hot (4),
BPP-diagonal unpaired prob (1)].  Edges = sequence adjacency + BPP pairing.
Candidates are folded with the reference aptamer as a fixed context so the
fold regime matches the measured FC- constructs.
"""
import numpy as np

from .config import BASE, BPP_THRESHOLD, CACHE_DIR
from utils.data_loader import construct_graph_from_bpp


def base_onehot(seq):
    return np.array([[1.0 if j == BASE.get(b, 0) else 0.0 for j in range(4)]
                     for b in seq], dtype=float)


def make_graph(core_seq, core_bpp):
    return construct_graph_from_bpp(base_onehot(core_seq), core_bpp,
                                    threshold=BPP_THRESHOLD)


def candidate_graph(core_seq, aptamer, cache_dir=CACHE_DIR):
    """Graph for a novel core: fold aptamer + core, take the candidate-region
    BPP submatrix (positions after the aptamer), build the graph."""
    from .data import get_bpp
    asm = aptamer + core_seq
    m = get_bpp(asm, cache_dir=cache_dir)
    L = len(core_seq)
    return make_graph(core_seq, m[70:70 + L, 70:70 + L])
