"""Data loading + ViennaRNA BPP (cached, parallel)."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import CACHE_DIR, N_JOBS, PISTOL_STO, RAW_CSV

_RNA = None


def _rna():
    global _RNA
    if _RNA is None:
        import RNA
        _RNA = RNA
    return _RNA


def compute_bpp(seq):
    """Free-fold base-pair probability matrix (N,N); diagonal = unpaired prob."""
    RNA = _rna()
    fc = RNA.fold_compound(seq)
    fc.pf()
    raw = fc.bpp()
    N = len(seq)
    m = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            m[i, j] = raw[i + 1][j + 1]
    m = m + m.T
    p = m.sum(axis=1)
    np.fill_diagonal(m, 1.0 - p)
    return m


def _cache_path(cache_dir, seq):
    return Path(cache_dir) / (hashlib.md5(seq.encode()).hexdigest()[:16] + ".npz")


def _load_cached(p):
    """Load a BPP cache entry; return None if missing/corrupted."""
    try:
        return np.load(p)["bpp"]
    except (EOFError, OSError, KeyError, ValueError):
        p.unlink(missing_ok=True)      # corrupted -> drop so it gets recomputed
        return None


def get_bpp(seq, cache_dir=None):
    if cache_dir:
        p = _cache_path(cache_dir, seq)
        cached = _load_cached(p) if p.exists() else None
        if cached is not None:
            return cached
        m = compute_bpp(seq)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, bpp=m)
        return m
    return compute_bpp(seq)


def compute_bpps(seqs, cache_dir=CACHE_DIR, j=N_JOBS):
    """Parallel + cached BPP for a list of sequences."""
    out = [None] * len(seqs)
    todo = []
    for i, s in enumerate(seqs):
        p = _cache_path(cache_dir, s) if cache_dir else None
        if cache_dir and p.exists():
            cached = _load_cached(p)
            if cached is not None:
                out[i] = cached
            else:
                todo.append(i)
        else:
            todo.append(i)
    if todo:
        res = Parallel(n_jobs=j, verbose=10, backend="multiprocessing")(
            delayed(get_bpp)(seqs[i], cache_dir) for i in todo)
        for i, m in zip(todo, res):
            out[i] = m
    return out


def load_rawdata():
    """Returns (list of 140-nt construct sequences, FC- label array)."""
    df = pd.read_csv(RAW_CSV)
    return df["RNAseq"].astype(str).tolist(), df["FC-"].to_numpy(float)


def load_pistol():
    """Returns the 676 real pistol cores (gaps stripped, canonical A/C/G/U)."""
    rows = {}
    for line in open(PISTOL_STO):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) >= 2:
            rows[parts[0]] = parts[1]
    clean = {"A": "A", "C": "C", "G": "G", "U": "U", "T": "U"}
    return ["".join(clean.get(b, "A") for b in s.replace(".", "").replace("-", "").upper())
            for s in rows.values()]


def load_fasta(path, limit=None):
    """Returns ungapped sequences (multi-line fasta handled)."""
    seqs, cur = [], []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur).upper().replace("T", "U"))
                cur = []
                if limit and len(seqs) >= limit:
                    break
        else:
            cur.append(line)
    if cur:
        seqs.append("".join(cur).upper().replace("T", "U"))
    return seqs


def load_csv_records(path, limit=0):
    """CSV -> list of (name, seq, slice_or_None).

    Columns: `seq` (or `sequence`) is required; `slice_start`/`slice_stop`
    (or `start`/`stop`) give the ribozyme-domain slice per sequence; `name` is
    optional.  Slices are 0-based python-style [start:stop].
    """
    df = pd.read_csv(path)
    seq_col = "seq" if "seq" in df.columns else ("sequence" if "sequence" in df.columns else None)
    if seq_col is None:
        raise ValueError(f"CSV needs a 'seq'/'sequence' column, got: {list(df.columns)}")
    start_col = next((c for c in ("slice_start", "start") if c in df.columns), None)
    stop_col = next((c for c in ("slice_stop", "stop") if c in df.columns), None)
    name_col = "name" if "name" in df.columns else None
    records = []
    for i, row in df.iterrows():
        if limit and i >= limit:
            break
        name = str(row[name_col]) if name_col else f"row{i}"
        seq = str(row[seq_col]).replace("T", "U").upper()
        sl = None
        if start_col and stop_col and pd.notna(row[start_col]) and pd.notna(row[stop_col]):
            sl = (int(row[start_col]), int(row[stop_col]))
        records.append((name, seq, sl))
    return records
