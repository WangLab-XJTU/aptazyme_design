"""Generate full-length constructs for the design scorer via pycmemit.

Pipeline per aptamer (from GuanidineAptamer.csv):
    aptamer tail of length L (8 or 16) -> pycmemit prefix
    -> pycmemit(RF02679.cm, prefix, N) emits cores that start with the prefix
    -> assemble full = aptamer + core[L:]   (the L-nt tail is shared: the core's
       5' head doubles as the aptamer's 3' tail)
    -> ribozyme-domain slice (0-based) = [A-L, A+len(core)-L], A = len(aptamer)

Parallel: (aptamer, prefix) units run in parallel via joblib (n_jobs=-1 = all
cores by default).  core_prob = P(core | CM), computed by re-emitting each core
as a prefix (N=1); unique cores are computed once per prefix.

Outputs:
    --out-design   scorer-input CSV:  name, seq (full-length), slice_start, slice_stop
    --out-full     process CSV with everything: aptamer id/len, L, prefix,
                   core, core_len, full seq, slice, prefix_prob, core_prob

Usage (from ribozyme/ root):
    python scripts/generator.py --max-aptamers 2                 # quick test
    python scripts/generator.py --max-aptamers 100 --n 200       # real run
    python scripts/generator.py -j 4 ...                         # override cores
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

RIB = Path(__file__).resolve().parents[1]
INF = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RIB))          # project root -> `utils.pycmemit`
try:
    from utils.pycmemit import emit   # pycmemit is a project utility
except ImportError:
    sys.path.insert(0, str(INF))      # fallback: parent dir pycmemit.py
    from pycmemit import emit
from scorer import config


def _core_prob(cm, core, seed):
    try:
        res = emit(cm, 1, prefix=core, seed=seed)
        return res["models"][0].get("prefix_prob")
    except RuntimeError:
        return None


def _process_prefix(apt_idx, apt_id, apt, L, cm, n, seed):
    """One (aptamer, prefix) unit -> (design_rows, full_rows, skip_or_None)."""
    prefix = apt[-L:]
    try:
        res = emit(cm, n, prefix=prefix, seed=seed)
    except RuntimeError as e:
        return [], [], {"apt_id": apt_id, "L": L, "prefix": prefix,
                        "prefix_prob": None, "error": str(e)}
    m = res["models"][0]
    cores = m["sequences"]
    pprob = m.get("prefix_prob")
    A = len(apt)
    core_prob_cache = {}
    design, full = [], []
    for i, core in enumerate(cores):
        core = core.replace("T", "U")
        cL = len(core)
        full_seq = apt + core[L:]                 # shared L-nt junction
        sstart, sstop = A - L, A + cL - L         # ribozyme-domain slice
        name = f"apt{apt_idx}_L{L}_{i}"
        if core not in core_prob_cache:
            core_prob_cache[core] = _core_prob(cm, core, seed)
        design.append({"name": name, "seq": full_seq,
                       "slice_start": sstart, "slice_stop": sstop})
        full.append({"name": name, "apt_id": apt_id, "apt_len": A, "L": L,
                     "prefix": prefix, "core": core, "core_len": cL,
                     "full_len": len(full_seq), "seq": full_seq,
                     "slice_start": sstart, "slice_stop": sstop,
                     "prefix_prob": pprob, "core_prob": core_prob_cache[core]})
    return design, full, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(RIB / "data" / "GuanidineAptamer.csv"))
    ap.add_argument("--cm", default=str(RIB / "data" / "RF02679.cm"))
    ap.add_argument("--n", type=int, default=200, help="sequences to emit per prefix")
    ap.add_argument("--prefix-lens", type=int, nargs="+", default=[8, 16],
                    help="aptamer tail lengths used as the prefix")
    ap.add_argument("--max-aptamers", type=int, default=2,
                    help="only process the first N aptamers (0 = all rows)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-j", "--jobs", type=int, default=config.N_JOBS,
                    help="parallel workers (-1 = all cores)")
    ap.add_argument("--out-design", default=str(RIB / "out" / "generated_design_input.csv"))
    ap.add_argument("--out-full", default=str(RIB / "out" / "generated_design_full.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.max_aptamers:
        df = df.head(args.max_aptamers)

    tasks = []
    for row_i, (_, r) in enumerate(df.iterrows()):
        apt_id = str(r.get("ID", f"apt{row_i}"))
        apt = str(r["seq"]).replace("-", "").replace(".", "").upper().replace("T", "U")
        for L in args.prefix_lens:
            tasks.append((row_i, apt_id, apt, L))

    print(f"aptamers: {len(df)}  prefixes: L={args.prefix_lens}  N={args.n}  "
          f"tasks: {len(tasks)}  jobs: {args.jobs}")

    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(_process_prefix)(i, aid, apt, L, str(args.cm), args.n, args.seed)
        for (i, aid, apt, L) in tasks)

    design_rows, full_rows, skipped = [], [], []
    for d, f, sk in results:
        design_rows.extend(d)
        full_rows.extend(f)
        if sk:
            skipped.append(sk)

    Path(args.out_design).parent.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame(design_rows)
    d.to_csv(args.out_design, index=False)
    f = pd.DataFrame(full_rows)
    f.to_csv(args.out_full, index=False)

    print(f"done: design={len(d)}  full={len(f)}  skipped={len(skipped)}")
    print(f"scorer input -> {args.out_design}")
    print(f"full info   -> {args.out_full}")
    if skipped:
        skp = Path(args.out_full).with_suffix(".skipped.csv")
        pd.DataFrame(skipped).to_csv(skp, index=False)
        print(f"skipped     -> {len(skipped)} (see {skp})")


if __name__ == "__main__":
    main()
