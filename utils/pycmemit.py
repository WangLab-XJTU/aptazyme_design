#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pycmemit.py
===========

A faithful Python reimplementation of the CM-emission path of Infernal
1.1.5's `cmemit` (the ``--nohmmonly`` behavior).

It reads Infernal 1.1.x ASCII covariance-model files ("INFERNAL1/a") and
generates sequences exactly as the C ``cmemit`` does, byte-for-byte for a
given RNG seed.  The port reproduces the exact arithmetic of the C code:

  * all probability vectors are IEEE-754 single precision (float32);
  * vector sums use the same Kahan algorithm as Easel;
  * the Easel "fast" LCG random number generator is emulated exactly
    (mix3 seed dispersion, Knuth LCG, Roll / FChoose / DChoose);
  * the ordering of random draws follows the C code exactly;
  * the integer/bool subtleties of the C code are preserved, including
    the stale-``y`` bug in ``createFaceCharts()``.

Validation: run it with ``--nohmmonly --seed 42`` on a 1.1.5-built
``cmemit`` and byte-diff the outputs.  The 25-case matrix in
``validate_cmemit.sh`` (FASTA, Stockholm, consensus, local, --exp, --u5p/
--u3p, -e/--iid embedding, --dna, --a5p/--a3p, --idx) passes byte-for-byte
on stdout, stderr and exit code; a further 13-case extended spot-check
(``_debug/extratests.sh``) also passes.

Known minor deviations (CLI only; emission output is unaffected):
  * mutually-incompatible option errors print a different message than
    Easel's ``Failed to parse command line: ...`` + full usage text
    (exit code still matches);
  * when ``-e <n>`` is too small for an emitted sequence, both sides abort
    with the same message; the partial output written before the abort is
    byte-identical.

Not implemented (raises NotImplementedError):
  * emitting from the filter HMM (``--hmmonly``; also the *default* path
    for models with zero basepairs, e.g. snR75, unless ``--nohmmonly``);
  * ``--tfile`` parsetree output;
  * ``--outformat`` other than Stockholm;
  * ``--seed 0`` (arbitrary one-time seed, not reproducible).

The ``--prefix <seq>`` feature draws each emitted sequence conditionally on
its 5' end matching <seq>.  The default ``forward`` sampler (``--sampler
forward``) computes inside-style DP tables ``pre``/``exact`` over the CM
state graph in O(P*M) and ancestral-samples a tree matching the prefix,
continuing unconditionally past it -- cost independent of P(prefix), so
even an extremely rare prefix (e.g. P ~ 1e-9) costs essentially nothing per
sequence.  An impossible prefix (probability exactly 0) fails immediately.
The old ``--sampler reject`` keeps early-screened rejection sampling
(~1/P(prefix) trials per sequence).  Draws are exact samples of
``P(seq, tree | seq[1:P] = prefix)`` either way, and the unconstrained path
(``--prefix`` not given) is byte-identical to before.  IUPAC ambiguity
codes (R,Y,M,K,S,W,H,B,V,D,N) and gap chars match any residue.

Requires: numpy (for exact float32 arithmetic).

Importable API
--------------
The same machinery is exposed as ``emit()`` for programmatic use::

    from pycmemit import emit
    res = emit("4.c.cm", N=50, prefix="GG", seed=42)
    print(res["models"][0]["sequences"])

``emit()`` validates its options like the CLI but raises ValueError /
NotImplementedError / RuntimeError instead of calling ``sys.exit``, and
returns a dict of generated sequences (plus metadata such as per-model
rejection-trial counts when ``--prefix`` is used).  See the function's
docstring for the full return schema.
"""

import math
import os
import sys
from io import StringIO

try:
    import numpy as np
except ImportError:                 # pragma: no cover
    sys.stderr.write("Error: pycmemit.py requires numpy (for float32 arithmetic).\n")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constants (mirrors infernal.h / cm.h)
# ---------------------------------------------------------------------------

# State types
D_st = 0; MP_st = 1; ML_st = 2; MR_st = 3
IL_st = 4; IR_st = 5; S_st = 6; E_st = 7; B_st = 8; EL_st = 9

# Node types
BIF_nd = 0; MATP_nd = 1; MATL_nd = 2; MATR_nd = 3
BEGL_nd = 4; BEGR_nd = 5; ROOT_nd = 6; END_nd = 7

# Unique state codes
ROOT_S = 0; ROOT_IL = 1; ROOT_IR = 2; BEGL_S = 3; BEGR_S = 4; BEGR_IL = 5
MATP_MP = 6; MATP_ML = 7; MATP_MR = 8; MATP_D = 9; MATP_IL = 10; MATP_IR = 11
MATL_ML = 12; MATL_D = 13; MATL_IL = 14; MATR_MR = 15; MATR_D = 16
MATR_IR = 17; END_E = 18; BIF_B = 19; END_EL = 20

# CM flags
CMH_BITS        = 1 << 0
CMH_RF          = 1 << 3
CMH_CHKSUM      = 1 << 7
CMH_MAP         = 1 << 8
CMH_CONS        = 1 << 9
CMH_LOCAL_BEGIN = 1 << 10
CMH_LOCAL_END   = 1 << 11

# CM config / align options
CM_CONFIG_LOCAL         = 1 << 0
CM_EMIT_NO_LOCAL_BEGINS = 1 << 21
CM_EMIT_NO_LOCAL_ENDS   = 1 << 22
CM_ALIGN_FLUSHINSERTS   = 1 << 16

# trace mode + parsetree
TRMODE_J        = 3
TRACE_LEFT_CHILD  = 1
TRACE_RIGHT_CHILD = 2
PDA_RESIDUE = 0
PDA_STATE   = 1
PDA_MARKER  = 2

# alignment
GAP = -1        # gap code (never emitted; gap chars are '-' / '.')

IMPOSSIBLE = -1e36
MAXCONNECT = 6
ESLDSQ_SENTINEL = 1

K = 4           # alphabet size (RNA/DNA), the only alphabets we support
RNA_SYM = "ACGU-RYMKSWHBVDN*~"
DNA_SYM = "ACGT-RYMKSWHBVDN*~"

INFERNAL_VERSION = "1.1.5"


# ---------------------------------------------------------------------------
# float32 helpers (bit-exact ports of Easel esl_vec_*)
# ---------------------------------------------------------------------------

def f32(x):
    """Cast a Python number to an IEEE-754 single-precision float32."""
    return np.float32(x)


def FSum(v):
    """esl_vec_FSum: Kahan-compensated sum in float32; returns float32."""
    sum = np.float32(0.0)
    c = np.float32(0.0)
    for vi in v:
        y = np.float32(vi) - c
        t = np.float32(sum + y)
        c = np.float32(np.float32(t - sum) - y)
        sum = t
    return sum


def DSum(v):
    """esl_vec_DSum: Kahan-compensated sum in double; returns Python float."""
    sum = 0.0
    c = 0.0
    for vi in v:
        y = vi - c
        t = sum + y
        c = (t - sum) - y
        sum = t
    return sum


def FNorm(v):
    """esl_vec_FNorm (in place): normalize float32 vector, Kahan sum."""
    n = len(v)
    sum = FSum(v)
    if float(sum) != 0.0:
        for i in range(n):
            v[i] = np.float32(v[i] / sum)
    else:
        for i in range(n):
            v[i] = np.float32(1.0 / n)


def DNorm(v):
    """esl_vec_DNorm (in place): normalize double vector, Kahan sum."""
    n = len(v)
    sum = DSum(v)
    if sum != 0.0:
        for i in range(n):
            v[i] = v[i] / sum
    else:
        for i in range(n):
            v[i] = 1.0 / n


def FScale(v, scale):
    """esl_vec_FScale (in place): v[i] *= scale; scale is float32."""
    for i in range(len(v)):
        v[i] = np.float32(v[i] * scale)


def FArgMax(v, n):
    """esl_vec_FArgMax: index of the first (strictly >) maximum."""
    best = 0
    for i in range(1, n):
        if float(v[i]) > float(v[best]):
            best = i
    return best


def sreLOG2(x):
    """sreLOG2 macro: log(x)*1.44269504, or IMPOSSIBLE for x<=0."""
    if x > 0.0:
        return math.log(x) * 1.44269504
    return IMPOSSIBLE


def sreEXP2(x):
    """sreEXP2 macro: exp(x*0.69314718)."""
    return math.exp(x * 0.69314718)


# ---------------------------------------------------------------------------
# Easel fast RNG (bit-exact port of easel/esl_random.c)
# ---------------------------------------------------------------------------

def esl_mix3(a, b, c):
    """esl_mix3(): the Bob Jenkins 96-bit mix, emulating uint32_t wrap."""
    a &= 0xFFFFFFFF; b &= 0xFFFFFFFF; c &= 0xFFFFFFFF
    a = (a - b) & 0xFFFFFFFF; a = (a - c) & 0xFFFFFFFF; a ^= (c >> 13)
    b = (b - c) & 0xFFFFFFFF; b = (b - a) & 0xFFFFFFFF; b ^= ((a << 8) & 0xFFFFFFFF)
    c = (c - a) & 0xFFFFFFFF; c = (c - b) & 0xFFFFFFFF; c ^= (b >> 13)
    a = (a - b) & 0xFFFFFFFF; a = (a - c) & 0xFFFFFFFF; a ^= (c >> 12)
    b = (b - c) & 0xFFFFFFFF; b = (b - a) & 0xFFFFFFFF; b ^= ((a << 16) & 0xFFFFFFFF)
    c = (c - a) & 0xFFFFFFFF; c = (c - b) & 0xFFFFFFFF; c ^= (b >> 5)
    a = (a - b) & 0xFFFFFFFF; a = (a - c) & 0xFFFFFFFF; a ^= (c >> 3)
    b = (b - c) & 0xFFFFFFFF; b = (b - a) & 0xFFFFFFFF; b ^= ((a << 10) & 0xFFFFFFFF)
    c = (c - a) & 0xFFFFFFFF; c = (c - b) & 0xFFFFFFFF; c ^= (b >> 15)
    return c & 0xFFFFFFFF


class Randomness:
    """esl_randomness with the fast (LCG) generator."""

    def __init__(self, seed):
        if seed == 0:
            raise NotImplementedError(
                "--seed 0 uses a one-time arbitrary seed and is not reproducible; "
                "pass an explicit seed (e.g. --seed 42)")
        self.x = esl_mix3(seed, 87654321, 12345678)
        if self.x == 0:
            self.x = 42

    def knuth(self):
        """esl_random_uint32: Knuth LCG."""
        self.x = (self.x * 69069 + 1) & 0xFFFFFFFF
        return self.x

    def random(self):
        """esl_random: x / 2^32 in [0,1)."""
        return self.knuth() / 4294967296.0

    def Roll(self, n):
        """esl_rnd_Roll: uniform [0,n-1] via rejection sampling."""
        factor = 0xFFFFFFFF // n
        while True:
            u = self.knuth() // factor
            if u < n:
                return u

    def FChoose(self, p, n):
        """esl_rnd_FChoose: draw from the first n floats of p (double math)."""
        norm = 0.0
        for i in range(n):
            norm += float(p[i])
        roll = self.random()
        sumv = 0.0
        for i in range(n):
            sumv += float(p[i])
            if roll < sumv / norm:
                return i
        raise RuntimeError("unreached code was reached. universe collapses.")

    def DChoose(self, p, n):
        """esl_rnd_DChoose: draw from the first n doubles of p."""
        norm = 0.0
        for i in range(n):
            norm += p[i]
        roll = self.random()
        sumv = 0.0
        for i in range(n):
            sumv += p[i]
            if roll < sumv / norm:
                return i
        raise RuntimeError("unreached code was reached. universe collapses.")


# ---------------------------------------------------------------------------
# CM parsing (read_asc_1p1_cm)
# ---------------------------------------------------------------------------

def ascii2prob(s, null):
    """ascii2prob(): (*s=='*') ? 0. : exp(atof(s)/1.44269504)*null, -> float32."""
    if s == '*':
        return np.float32(0.0)
    return np.float32(math.exp(float(s) / 1.44269504) * float(null))


def StateCode(s):
    if   s == "D":  return D_st
    elif s == "MP": return MP_st
    elif s == "ML": return ML_st
    elif s == "MR": return MR_st
    elif s == "IL": return IL_st
    elif s == "IR": return IR_st
    elif s == "S":  return S_st
    elif s == "E":  return E_st
    elif s == "B":  return B_st
    elif s == "EL": return EL_st
    return -1


def NodeCode(s):
    if   s == "BIF":  return BIF_nd
    elif s == "MATP": return MATP_nd
    elif s == "MATL": return MATL_nd
    elif s == "MATR": return MATR_nd
    elif s == "BEGL": return BEGL_nd
    elif s == "BEGR": return BEGR_nd
    elif s == "ROOT": return ROOT_nd
    elif s == "END":  return END_nd
    return -1


def DeriveUniqueStateCode(ndtype, sttype):
    """DeriveUniqueStateCode() in cm.c."""
    if ndtype == BIF_nd:
        if sttype == B_st: return BIF_B
        return -1
    elif ndtype == MATP_nd:
        if sttype == D_st:  return MATP_D
        if sttype == MP_st: return MATP_MP
        if sttype == ML_st: return MATP_ML
        if sttype == MR_st: return MATP_MR
        if sttype == IL_st: return MATP_IL
        if sttype == IR_st: return MATP_IR
        return -1
    elif ndtype == MATL_nd:
        if sttype == D_st:  return MATL_D
        if sttype == ML_st: return MATL_ML
        if sttype == IL_st: return MATL_IL
        return -1
    elif ndtype == MATR_nd:
        if sttype == D_st:  return MATR_D
        if sttype == MR_st: return MATR_MR
        if sttype == IR_st: return MATR_IR
        return -1
    elif ndtype == BEGL_nd:
        if sttype == S_st: return BEGL_S
        return -1
    elif ndtype == BEGR_nd:
        if sttype == S_st:  return BEGR_S
        if sttype == IL_st: return BEGR_IL
        return -1
    elif ndtype == ROOT_nd:
        if sttype == S_st:  return ROOT_S
        if sttype == IL_st: return ROOT_IL
        if sttype == IR_st: return ROOT_IR
        return -1
    elif ndtype == END_nd:
        if sttype == E_st: return END_E
        return -1
    return -1


class CM:
    """A parsed covariance model (the fields cmemit needs)."""
    __slots__ = ("name", "acc", "desc", "M", "nodes", "clen", "W", "flags",
                 "pbegin", "pend", "el_selfsc", "null",
                 "sttype", "plast", "pnum", "cfirst", "cnum",
                 "ndidx", "stid", "nodemap", "ndtype",
                 "t", "e", "esc", "begin", "end", "root_trans",
                 "emap", "cmcons", "config_opts", "align_opts", "abc_type",
                 "rf")

    def __init__(self):
        self.name = None
        self.acc = None
        self.desc = None
        self.M = 0
        self.nodes = 0
        self.clen = 0
        self.W = 0
        self.flags = 0
        self.pbegin = np.float32(0.0)
        self.pend = np.float32(0.0)
        self.el_selfsc = np.float32(0.0)
        self.null = []
        self.sttype = []
        self.plast = []
        self.pnum = []
        self.cfirst = []
        self.cnum = []
        self.ndidx = []
        self.stid = []
        self.nodemap = []
        self.ndtype = []
        self.t = []
        self.e = []
        self.esc = []
        self.begin = None
        self.end = None
        self.root_trans = None
        self.emap = None
        self.cmcons = None
        self.config_opts = 0
        self.align_opts = 0
        self.abc_type = "RNA"
        self.rf = None


def read_cm_file(path):
    """Read all CMs from a .cm file. Returns list of CM."""
    with open(path, "rb") as f:
        data = f.read().decode("latin-1")
    lines = data.split("\n")
    li = 0
    nlines = len(lines)
    cms = []
    while li < nlines:
        toks = lines[li].split()
        if not toks:
            li += 1
            continue
        if toks[0] == "INFERNAL1/a":
            cm, li = _read_one_cm(lines, li + 1)
            cms.append(cm)
        else:
            li += 1
    return cms


def _read_one_cm(lines, li):
    cm = CM()
    nlines = len(lines)

    # ---- header ----
    while True:
        if li >= nlines:
            raise RuntimeError("premature EOF in CM header")
        toks = lines[li].split()
        li += 1
        if not toks:
            continue
        tag = toks[0]
        if tag == "NAME":
            cm.name = toks[1]
        elif tag == "ACC":
            cm.acc = toks[1]
        elif tag == "DESC":
            cm.desc = toks[1]
        elif tag == "STATES":
            cm.M = int(toks[1])
        elif tag == "NODES":
            cm.nodes = int(toks[1])
        elif tag == "CLEN":
            pass                        # recomputed from the model
        elif tag == "W":
            cm.W = int(toks[1])
        elif tag == "ALPH":
            atype = toks[1].upper()
            if atype not in ("RNA", "DNA"):
                raise RuntimeError("unsupported alphabet %s" % toks[1])
            cm.abc_type = atype
        elif tag == "PBEGIN":
            cm.pbegin = np.float32(float(toks[1]))
        elif tag == "PEND":
            cm.pend = np.float32(float(toks[1]))
        elif tag == "ELSELF":
            cm.el_selfsc = np.float32(float(toks[1]))
        elif tag == "NULL":
            cm.null = [ascii2prob(toks[1 + x], 1.0 / float(K)) for x in range(K)]
        elif tag == "RF":
            if toks[1].lower() == "yes":
                cm.flags |= CMH_RF
        elif tag == "CONS":
            if toks[1].lower() == "yes":
                cm.flags |= CMH_CONS
        elif tag == "MAP":
            if toks[1].lower() == "yes":
                cm.flags |= CMH_MAP
        elif tag == "CKSUM":
            cm.flags |= CMH_CHKSUM
        elif tag == "CM":
            break                       # end of header
        # any other tag (DATE, COM, NSEQ, EFFN, GA, TC, NC, EFP7GF, ECM,
        # WBETA, QDBBETA1, QDBBETA2, N2OMEGA, N3OMEGA, ...) is ignored

    # ---- allocate ----
    M = cm.M
    cm.sttype  = [0] * M
    cm.plast   = [0] * M
    cm.pnum    = [0] * M
    cm.cfirst  = [0] * M
    cm.cnum    = [0] * M
    cm.ndidx   = [0] * M
    cm.stid    = [-1] * M
    cm.nodemap = [-1] * cm.nodes
    cm.ndtype  = [-1] * cm.nodes
    cm.t       = [None] * M
    cm.e       = [None] * M
    cm.esc     = [[] for _ in range(M)]
    cm.clen    = 0
    cm.rf      = None
    tmp_rf_left  = ['-'] * cm.nodes
    tmp_rf_right = ['-'] * cm.nodes

    # ---- states ----
    v = 0
    nd = -1
    while v < M:
        while True:
            toks = lines[li].split()
            if toks:
                break
            li += 1
        if toks[0] == '[':
            # node line: [ <ntype> <nd> ] <mapL> <mapR> <consL> <consR> <rfL> <rfR>
            ntype = NodeCode(toks[1])
            nd = int(toks[2])
            cm.ndtype[nd] = ntype
            if ntype == MATP_nd:
                cm.clen += 2
            elif ntype == MATL_nd or ntype == MATR_nd:
                cm.clen += 1
            cm.nodemap[nd] = v
            tmp_rf_left[nd] = toks[8][0]
            tmp_rf_right[nd] = toks[9][0]
            li += 1
            toks = lines[li].split()
        # state line
        cm.sttype[v]  = StateCode(toks[0])
        cm.plast[v]   = int(toks[2])
        cm.pnum[v]    = int(toks[3])
        cm.cfirst[v]  = int(toks[4])
        cm.cnum[v]    = int(toks[5])
        idx = 10
        if cm.sttype[v] != B_st:
            cm.t[v] = [ascii2prob(toks[idx + x], 1.0) for x in range(cm.cnum[v])]
        else:
            cm.t[v] = []
        eidx = idx + cm.cnum[v]
        if cm.sttype[v] in (ML_st, MR_st, IL_st, IR_st):
            cm.e[v] = [ascii2prob(toks[eidx + x], float(cm.null[x])) for x in range(K)]
        elif cm.sttype[v] == MP_st:
            cm.e[v] = [ascii2prob(toks[eidx + x],
                                  float(cm.null[x // K]) * float(cm.null[x % K]))
                       for x in range(K * K)]
        else:
            cm.e[v] = []
        cm.ndidx[v] = nd
        cm.stid[v]  = DeriveUniqueStateCode(cm.ndtype[nd], cm.sttype[v])
        v += 1
        li += 1

    # the EL state at M is special: only state-type info is recorded, so
    # parsetree consumers can interpret what an "M" index means (cm.c CMCreate).
    cm.sttype.append(EL_st)          # cm->sttype[cm->M] = EL_st
    cm.stid.append(END_EL)           # cm->stid[cm->M]   = END_EL

    # ---- closing // ----
    toks = lines[li].split()
    if not toks or toks[0] != "//":
        raise RuntimeError("Expected closing // after states")
    li += 1

    # ---- skip the filter-HMM block (until its own //) ----
    while li < nlines:
        toks = lines[li].split()
        if toks and toks[0] == "//":
            break
        li += 1
    li += 1

    # what read_asc_1p1_cm does after parsing:
    CMRenormalize(cm)
    cm.emap = CreateEmitMap(cm)
    if cm.flags & CMH_RF:
        # map per-node RF annotations to per-consensus-position rf string
        rf = [' '] * (cm.clen + 2)
        for nd in range(cm.nodes):
            if cm.ndtype[nd] == MATP_nd or cm.ndtype[nd] == MATL_nd:
                rf[cm.emap.lpos[nd]] = tmp_rf_left[nd]
            if cm.ndtype[nd] == MATP_nd or cm.ndtype[nd] == MATR_nd:
                rf[cm.emap.rpos[nd]] = tmp_rf_right[nd]
        cm.rf = rf
    return cm, li


# ---------------------------------------------------------------------------
# Model configuration (cm.c / cm_modelconfig.c / display.c)
# ---------------------------------------------------------------------------

def CMRenormalize(cm):
    """CMRenormalize(): FNorm all probability vectors."""
    FNorm(cm.null)
    for v in range(cm.M):
        if cm.cnum[v] > 0 and cm.sttype[v] != B_st:
            FNorm(cm.t[v])
        st = cm.sttype[v]
        if st == MP_st or st == ML_st or st == MR_st or st == IL_st or st == IR_st:
            FNorm(cm.e[v])
    if cm.flags & CMH_LOCAL_BEGIN:
        FNorm(cm.begin)
    if cm.flags & CMH_LOCAL_END:
        cm_Fail("CMRenormalize() model is in local mode")   # never happens


def cm_Exponentiate(cm, z):
    """cm_Exponentiate(): raise all transition/emission probs to power z."""
    if cm.flags & (CMH_LOCAL_BEGIN | CMH_LOCAL_END):
        cm_Fail("cm_Exponentiate() model is not in global configuration")
    for v in range(cm.M):
        if cm.sttype[v] != B_st and cm.sttype[v] != E_st:
            for x in range(cm.cnum[v]):
                cm.t[v][x] = np.float32(math.pow(float(cm.t[v][x]), z))
        if cm.sttype[v] == MP_st:
            for x in range(K * K):
                cm.e[v][x] = np.float32(math.pow(float(cm.e[v][x]), z))
        elif cm.sttype[v] in (ML_st, MR_st, IL_st, IR_st):
            for x in range(K):
                cm.e[v][x] = np.float32(math.pow(float(cm.e[v][x]), z))
    CMRenormalize(cm)
    cm.flags &= ~CMH_BITS


def cm_CalculateLocalBeginProbs(cm, p_internal_start):
    """cm_CalculateLocalBeginProbs(): compute the local begin probs."""
    nstarts = 0
    for nd in range(2, cm.nodes):
        if cm.ndtype[nd] in (MATP_nd, MATL_nd, MATR_nd, BIF_nd):
            nstarts += 1
    begin = [np.float32(0.0)] * cm.M
    begin[cm.nodemap[1]] = np.float32(1.0 - float(p_internal_start))
    p = np.float32(np.float32(p_internal_start) / np.float32(nstarts))
    for nd in range(2, cm.nodes):
        if cm.ndtype[nd] in (MATP_nd, MATL_nd, MATR_nd, BIF_nd):
            begin[cm.nodemap[nd]] = p
    return begin


def cm_localize(cm, p_internal_start, p_internal_exit):
    """cm_localize(): put the CM into local configuration."""
    cm.begin = cm_CalculateLocalBeginProbs(cm, p_internal_start)
    if cm.root_trans is None:
        cm.root_trans = [np.float32(x) for x in cm.t[0][:cm.cnum[0]]]   # FCopy
    cm.t[0] = [np.float32(0.0)] * cm.cnum[0]                            # FSet 0
    cm.flags |= CMH_LOCAL_BEGIN

    nexits = 0
    for nd in range(1, cm.nodes):
        if (cm.ndtype[nd] in (MATP_nd, MATL_nd, MATR_nd, BEGL_nd, BEGR_nd)
                and cm.ndtype[nd + 1] != END_nd):
            nexits += 1
    cm.end = [np.float32(0.0)] * cm.M
    for nd in range(1, cm.nodes):
        if (cm.ndtype[nd] in (MATP_nd, MATL_nd, MATR_nd, BEGL_nd, BEGR_nd)
                and cm.ndtype[nd + 1] != END_nd):
            v = cm.nodemap[nd]
            cm.end[v] = np.float32(np.float32(p_internal_exit) / np.float32(nexits))
            denom = FSum(cm.t[v])
            denom = np.float32(denom + cm.end[v])
            scale = np.float32(1.0 / float(denom))
            FScale(cm.t[v], scale)
    cm.flags |= CMH_LOCAL_END
    cm.flags &= ~CMH_BITS


def CMLogoddsify(cm):
    """CMLogoddsify(): compute esc (log-odds) scores, then cmcons."""
    for v in range(cm.M):
        st = cm.sttype[v]
        if st == MP_st:
            e = []
            for x in range(K * K):
                q = np.float32(cm.e[v][x] / np.float32(cm.null[x // K] * cm.null[x % K]))
                e.append(np.float32(sreLOG2(float(q))))
            cm.esc[v] = e
        elif st in (ML_st, MR_st, IL_st, IR_st):
            e = []
            for x in range(K):
                q = np.float32(cm.e[v][x] / cm.null[x])
                e.append(np.float32(sreLOG2(float(q))))
            cm.esc[v] = e
    cm.flags |= CMH_BITS
    cm.cmcons = CreateCMConsensus(cm)


def createMultifurcationOrderChart(cm):
    """createMultifurcationOrderChart(): multifurcation order per node."""
    height = [0] * cm.nodes
    seg_has_pairs = [0] * cm.nodes
    for nd in range(cm.nodes - 1, -1, -1):
        v = cm.nodemap[nd]
        if   cm.stid[v] == MATP_MP: seg_has_pairs[nd] = True
        elif cm.stid[v] == END_E:   seg_has_pairs[nd] = False
        elif cm.stid[v] == BIF_B:   seg_has_pairs[nd] = False
        else:                       seg_has_pairs[nd] = seg_has_pairs[nd + 1]
        if cm.stid[v] == END_E:
            height[nd] = 0
        elif cm.stid[v] == BIF_B:
            left  = cm.ndidx[cm.cfirst[v]]
            right = cm.ndidx[cm.cnum[v]]
            height[nd] = max(height[left] + (1 if seg_has_pairs[left] else 0),
                             height[right] + (1 if seg_has_pairs[right] else 0))
        else:
            height[nd] = height[nd + 1]
    return height


def createFaceCharts(cm):
    """createFaceCharts(): inface/outface per node.  Faithfully preserves the
    stale-`y` bug in the BEGR branch of the C code."""
    inface = [0] * cm.nodes
    for nd in range(cm.nodes - 1, -1, -1):
        v = cm.nodemap[nd]
        if cm.ndtype[nd] == END_nd:
            inface[nd] = 0
        elif cm.ndtype[nd] == BIF_nd:
            left  = cm.ndidx[cm.cfirst[v]]
            right = cm.ndidx[cm.cnum[v]]
            inface[nd] = inface[left] + inface[right]
        else:
            if cm.ndtype[nd + 1] == MATP_nd:
                inface[nd] = 1
            else:
                inface[nd] = inface[nd + 1]

    outface = [0] * cm.nodes
    y = 0   # NOTE: y is only assigned in the BEGL branch; BEGR uses the stale y
    for nd in range(cm.nodes):
        v = cm.nodemap[nd]
        if cm.ndtype[nd] == ROOT_nd:
            outface[nd] = 0
        elif cm.ndtype[nd] == BEGL_nd:
            parent = cm.ndidx[cm.plast[v]]
            y      = cm.nodemap[parent]
            right  = cm.ndidx[cm.cnum[y]]
            outface[nd] = outface[parent] + inface[right]
        elif cm.ndtype[nd] == BEGR_nd:
            parent = cm.ndidx[cm.plast[v]]
            left   = cm.ndidx[cm.cfirst[y]]     # stale y
            outface[nd] = outface[parent] + inface[left]
        else:
            parent = nd - 1
            if cm.ndtype[parent] == MATP_nd:
                outface[nd] = 1
            else:
                outface[nd] = outface[parent]
    return inface, outface


class EmitMap:
    __slots__ = ("lpos", "rpos", "epos", "clen")

    def __init__(self, nodes):
        self.lpos = [-1] * nodes
        self.rpos = [-1] * nodes
        self.epos = [-1] * nodes
        self.clen = 0


def CreateEmitMap(cm):
    """CreateEmitMap(): consensus-position maps from the model architecture."""
    nodes = cm.nodes
    m = EmitMap(nodes)
    cpos = 0
    pda = []
    pda.append(0)   # on_right = 0 (left side)
    pda.append(0)   # nd
    while pda:
        nd = pda.pop()
        on_right = pda.pop()
        if on_right:
            m.rpos[nd] = cpos + 1
            if cm.ndtype[nd] == MATP_nd or cm.ndtype[nd] == MATR_nd:
                cpos += 1
        else:
            if cm.ndtype[nd] == MATP_nd or cm.ndtype[nd] == MATL_nd:
                cpos += 1
            m.lpos[nd] = cpos
            if cm.ndtype[nd] == BIF_nd:
                pda.append(1); pda.append(nd)                       # BIF right side
                pda.append(0); pda.append(cm.ndidx[cm.cnum[cm.nodemap[nd]]])   # right child
                pda.append(0); pda.append(cm.ndidx[cm.cfirst[cm.nodemap[nd]]]) # left child
            else:
                pda.append(1); pda.append(nd)
                if cm.ndtype[nd] != END_nd:
                    pda.append(0); pda.append(nd + 1)
    # epos
    for nd in range(nodes - 1, -1, -1):
        if cm.ndtype[nd] == END_nd:
            cpos = m.lpos[nd]
        elif cm.ndtype[nd] == BIF_nd:
            cpos = m.epos[cm.ndidx[cm.cnum[cm.nodemap[nd]]]]
        m.epos[nd] = cpos
    m.clen = m.rpos[0] - 1
    return m


class CMConsensus:
    __slots__ = ("cseq", "cstr", "ct", "lpos", "rpos", "clen")

    def __init__(self):
        self.cseq = []
        self.cstr = []
        self.ct = []
        self.lpos = []
        self.rpos = []
        self.clen = 0


def CreateCMConsensus(cm):
    """CreateCMConsensus(): consensus sequence/structure from esc scores."""
    if not (cm.flags & CMH_BITS):
        return None
    nodes = cm.nodes
    con = CMConsensus()
    lpos = [-1] * nodes
    rpos = [-1] * nodes
    multiorder = createMultifurcationOrderChart(cm)
    inface, outface = createFaceCharts(cm)
    abc_sym = RNA_SYM if cm.abc_type == "RNA" else DNA_SYM
    cseq = []
    cstr = []
    ct = []
    pda = []
    cpos = 0
    pda.append(0)
    pda.append(PDA_STATE)
    while pda:
        typ = pda.pop()
        if typ == PDA_RESIDUE:
            rchar = chr(pda.pop())
            rstruc = chr(pda.pop())
            pairpartner = pda.pop()
            nd = pda.pop()
            rpos[nd] = cpos
            cseq.append(rchar)
            cstr.append(rstruc)
            ct.append(pairpartner)
            if pairpartner != -1:
                ct[pairpartner] = cpos
            cpos += 1
        elif typ == PDA_MARKER:
            nd = pda.pop()
            rpos[nd] = cpos - 1
        else:   # PDA_STATE
            v = pda.pop()
            nd = cm.ndidx[v]
            lchar = rchar = lstruc = rstruc = ''
            if cm.stid[v] == MATP_MP:
                x = FArgMax(cm.esc[v], K * K)
                lchar = abc_sym[x // K]
                rchar = abc_sym[x % K]
                if float(cm.esc[v][x]) < 3.0:
                    lchar = lchar.lower()
                    rchar = rchar.lower()
                mo = multiorder[nd]
                if   mo == 0: lstruc = '<'; rstruc = '>'
                elif mo == 1: lstruc = '('; rstruc = ')'
                elif mo == 2: lstruc = '['; rstruc = ']'
                else:         lstruc = '{'; rstruc = '}'
            elif cm.stid[v] == MATL_ML:
                x = FArgMax(cm.esc[v], K)
                lchar = abc_sym[x]
                if float(cm.esc[v][x]) < 1.0:
                    lchar = lchar.lower()
                if   outface[nd] == 0:                    lstruc = ':'
                elif inface[nd] == 0 and outface[nd] == 1: lstruc = '_'
                elif inface[nd] == 1 and outface[nd] == 1: lstruc = '-'
                else:                                      lstruc = ','
                rstruc = ' '
            elif cm.stid[v] == MATR_MR:
                x = FArgMax(cm.esc[v], K)
                rchar = abc_sym[x]
                if float(cm.esc[v][x]) < 1.0:
                    rchar = rchar.lower()
                if   outface[nd] == 0:                    rstruc = ':'
                elif inface[nd] == 0 and outface[nd] == 1: rstruc = '?'
                elif inface[nd] == 1 and outface[nd] == 1: rstruc = '-'
                else:                                      rstruc = ','
                lstruc = ' '

            lpos[nd] = cpos
            if lchar:
                cseq.append(lchar)
                cstr.append(lstruc)
                ct.append(-1)
                cpos += 1
            if rchar:
                pda.append(nd)
                pda.append(cpos - 1 if lchar else -1)
                pda.append(ord(rstruc))
                pda.append(ord(rchar))
                pda.append(PDA_RESIDUE)
            else:
                pda.append(nd)
                pda.append(PDA_MARKER)

            if cm.sttype[v] == B_st:
                pda.append(cm.cnum[v]); pda.append(PDA_STATE)     # right S
                pda.append(cm.cfirst[v]); pda.append(PDA_STATE)   # left S
            elif cm.sttype[v] != E_st:
                v2 = cm.nodemap[cm.ndidx[cm.cfirst[v] + cm.cnum[v] - 1]]
                pda.append(v2); pda.append(PDA_STATE)

    con.cseq = "".join(cseq)
    con.cstr = "".join(cstr)
    con.ct = ct
    con.lpos = lpos
    con.rpos = rpos
    con.clen = cpos
    return con


def cm_Configure(cm):
    """cm_ConfigureSub() with W_from_cmdline=-1 (as cmemit calls it), reduced
    to the steps that affect CM emission."""
    # cm_Validate / cm_nonconfigured_Verify pass for valid .cm files.
    if cm.emap is None:
        cm.emap = CreateEmitMap(cm)
    if cm.config_opts & CM_CONFIG_LOCAL:
        cm_localize(cm, cm.pbegin, cm.pend)
    if float(cm.el_selfsc) * float(cm.W) < IMPOSSIBLE:
        cm.el_selfsc = np.float32(IMPOSSIBLE / (cm.W + 1))
    CMLogoddsify(cm)


# ---------------------------------------------------------------------------
# Parsetree construction (cm_parsetree.c)
# ---------------------------------------------------------------------------

class Parsetree:
    __slots__ = ("n", "state", "emitl", "emitr", "mode", "nxtl", "nxtr", "prv")

    def __init__(self):
        self.n = 0
        self.state = []
        self.emitl = []
        self.emitr = []
        self.mode = []
        self.nxtl = []
        self.nxtr = []
        self.prv = []


def InsertTraceNodewithMode(tr, y, whichway, emitl, emitr, state, mode):
    """InsertTraceNodewithMode() in cm_parsetree.c."""
    if y >= 0:
        a = tr.nxtl[y] if whichway == TRACE_LEFT_CHILD else tr.nxtr[y]
    else:
        a = -1
    n = tr.n
    tr.emitl.append(emitl)
    tr.emitr.append(emitr)
    tr.state.append(state)
    tr.mode.append(mode)
    tr.nxtl.append(a)
    tr.nxtr.append(-1)
    tr.prv.append(y)
    if y >= 0:
        if whichway == TRACE_LEFT_CHILD:
            tr.nxtl[y] = n
        else:
            tr.nxtr[y] = n
    if a != -1:
        tr.prv[a] = n
    tr.n += 1
    return n


def InsertTraceNode(tr, y, whichway, emitl, emitr, state):
    return InsertTraceNodewithMode(tr, y, whichway, emitl, emitr, state, TRMODE_J)


def ModeEmitsLeft(mode):
    if mode == TRMODE_J:
        return True
    if mode == 1:      # TRMODE_L
        return True
    return False


def ModeEmitsRight(mode):
    if mode == TRMODE_J:
        return True
    if mode == 2:      # TRMODE_R
        return True
    return False


class Seq:
    """A digital sequence (codes[0] = sentinel; codes[1..n] are residues)."""
    __slots__ = ("name", "codes", "n")

    def __init__(self, name, codes_1based):
        self.name = name
        self.codes = codes_1based
        self.n = len(codes_1based) - 1


class PrefixMismatch(Exception):
    """Raised by EmitParsetree() when a prefix-constrained emission deviates
    from the requested prefix.  Used to early-abort a rejection-sampling
    trial as soon as the acceptance predicate is known to be false."""


def EmitParsetree(cm, rng, name, prefix=None):
    """EmitParsetree(): generate one sequence + parsetree from a configured CM.

    Returns (tr, seq) where seq is a Seq of digital codes.

    If ``prefix`` is a tuple of residue codes (ints in 0..K-1, or None for a
    wildcard that matches any residue), the emitted 5' sequence is screened
    as it is generated: on the first emitted residue that cannot match the
    prefix, PrefixMismatch is raised (so a caller can early-abort a
    rejection-sampling trial).  A completed walk shorter than the prefix is
    also rejected.  ``prefix=None`` (the default) is a strict no-op: the
    function is byte-identical to the unconstrained path for a given seed.
    """
    if (cm.flags & CMH_LOCAL_END
            and abs(sreEXP2(float(cm.el_selfsc)) - 1.0) < 0.01):
        cm_Fail("EL self transition probability %f is too high, would emit "
                "long (too long) EL insertions." % sreEXP2(float(cm.el_selfsc)))

    P = 0 if prefix is None else len(prefix)   # prefix length to screen

    tr = Parsetree()
    gsq = []
    N = 0
    tmp_tvec = [np.float32(0.0)] * (MAXCONNECT + 1)

    pda = []
    pda.append(-1)                              # rchar
    pda.append(-1)                              # lchar
    pda.append(TRACE_LEFT_CHILD)                # whichway
    pda.append(-1)                              # tparent
    pda.append(0)                               # v
    pda.append(PDA_STATE)

    while pda:
        typ = pda.pop()
        if typ == PDA_RESIDUE:
            tpos = pda.pop()
            rchar = pda.pop()
            if rchar != -1:
                gsq.append(rchar)
                N += 1
                if P and N <= P and prefix[N - 1] is not None \
                        and rchar != prefix[N - 1]:
                    raise PrefixMismatch
            tr.emitr[tpos] = N
        else:   # PDA_STATE
            v = pda.pop()
            tparent = pda.pop()
            whichway = pda.pop()
            lchar = pda.pop()
            rchar = pda.pop()
            tpos = InsertTraceNode(tr, tparent, whichway, N + 1, -1, v)
            if lchar != -1:
                gsq.append(lchar)
                N += 1
                if P and N <= P and prefix[N - 1] is not None \
                        and lchar != prefix[N - 1]:
                    raise PrefixMismatch
            pda.append(rchar)
            pda.append(tpos)
            pda.append(PDA_RESIDUE)

            if cm.sttype[v] == B_st:
                y = cm.cfirst[v]
                z = cm.cnum[v]
                # push right child, then left child (left is popped first)
                pda.append(-1); pda.append(-1); pda.append(TRACE_RIGHT_CHILD)
                pda.append(tpos); pda.append(z); pda.append(PDA_STATE)
                pda.append(-1); pda.append(-1); pda.append(TRACE_LEFT_CHILD)
                pda.append(tpos); pda.append(y); pda.append(PDA_STATE)
            else:
                if v == 0 and cm.flags & CMH_LOCAL_BEGIN:
                    if cm.flags & CM_EMIT_NO_LOCAL_BEGINS:
                        y = cm.cfirst[v] + rng.FChoose(cm.root_trans, cm.cnum[0])
                    else:
                        y = rng.FChoose(cm.begin, cm.M)
                elif cm.flags & CMH_LOCAL_END:
                    if cm.flags & CM_EMIT_NO_LOCAL_ENDS:
                        tv = [np.float32(x) for x in cm.t[v]]
                        FNorm(tv)
                        y = cm.cfirst[v] + rng.FChoose(tv, cm.cnum[v])
                    else:
                        tv = [np.float32(x) for x in cm.t[v]]
                        tv.append(cm.end[v])
                        y = rng.FChoose(tv, cm.cnum[v] + 1)
                        if y == cm.cnum[v]:
                            y = cm.M
                        else:
                            y += cm.cfirst[v]
                else:
                    y = cm.cfirst[v] + rng.FChoose(cm.t[v], cm.cnum[v])

                st = cm.sttype[y]
                if st == MP_st:
                    x = rng.FChoose(cm.e[y], K * K)
                    lchar = x // K
                    rchar = x % K
                elif st == ML_st or st == IL_st:
                    lchar = rng.FChoose(cm.e[y], K)
                    rchar = -1
                elif st == MR_st or st == IR_st:
                    lchar = -1
                    rchar = rng.FChoose(cm.e[y], K)
                else:   # EL_st, E_st, S_st, D_st, B_st
                    lchar = -1
                    rchar = -1

                if st == E_st:
                    InsertTraceNode(tr, tpos, TRACE_LEFT_CHILD, N + 1, N, y)
                elif st == EL_st:   # y == cm->M
                    lpos = N + 1
                    tmp_tvec[0] = np.float32(sreEXP2(float(cm.el_selfsc)))
                    tmp_tvec[1] = np.float32(1.0 - float(tmp_tvec[0]))
                    y = rng.FChoose(tmp_tvec[:2], 2)
                    while y == 0:
                        lchar = rng.FChoose(cm.null, K)
                        gsq.append(lchar)
                        N += 1
                        if P and N <= P and prefix[N - 1] is not None \
                                and lchar != prefix[N - 1]:
                            raise PrefixMismatch
                        y = rng.FChoose(tmp_tvec[:2], 2)
                    InsertTraceNode(tr, tpos, TRACE_LEFT_CHILD, lpos, N, cm.M)
                else:
                    pda.append(rchar)
                    pda.append(lchar)
                    pda.append(TRACE_LEFT_CHILD)
                    pda.append(tpos)
                    pda.append(y)
                    pda.append(PDA_STATE)

    if P and N < P:
        raise PrefixMismatch

    return tr, Seq(name, [ESLDSQ_SENTINEL] + gsq)


def str2prefix(prefix, abc_sym):
    """Convert a --prefix string into a screening tuple for EmitParsetree.

    Each character maps to its digital residue code (A/C/G/U -> 0..3, or
    A/C/G/T for DNA).  IUPAC ambiguity codes and gap chars are wildcards
    (None) that match any emitted residue.  Unknown characters are fatal.
    """
    idx = {}
    for i in range(K):
        idx[abc_sym[i]] = i
    codes = []
    for ch in prefix.upper():
        if ch in idx:
            codes.append(idx[ch])
        elif ch in "RYMKSWHBVDN-.":
            codes.append(None)                  # ambiguity/gap: matches anything
        else:
            cm_Fail("unrecognized residue '%s' in --prefix '%s'"
                    % (ch, prefix))
    return tuple(codes)


def prefix_tostr(prefix, abc_sym=RNA_SYM):
    """Human-readable form of a screening tuple (for error messages)."""
    return "".join(abc_sym[c] if c is not None else 'N' for c in prefix)


def sample_one_with_prefix(cm, rng, prefix, name, max_trials=None):
    """Draw one sequence whose emitted 5' end matches ``prefix``.

    Early-screened rejection sampling: repeatedly emit from the plain CM,
    accepting the first trial whose emitted 5' sequence matches ``prefix``.
    Because EmitParsetree aborts a trial at the first mismatching residue,
    each trial costs only O(E[#residues]) draws instead of a full emission.
    The accepted draw is an exact sample of the conditional distribution
    P(seq, tree | seq[1:P] = prefix); deterministic for a fixed seed.

    Unless ``max_trials`` is given, the search runs until a sequence is
    accepted (bounded only by wall time / a user interrupt).  When
    ``max_trials`` is given, it aborts with cm_Fail once exceeded.

    Returns ``(tr, seq, ntrials)``: the parsetree, the accepted sequence,
    and the number of rejection trials it took (>= 1).
    """
    trial = 0
    while True:
        trial += 1
        if max_trials is not None and trial > max_trials:
            cm_Fail("--prefix '%s': no sequence accepted in %d trials; the "
                    "prefix is unlikely or impossible under this CM (raise "
                    "--max-trials)" % (prefix_tostr(prefix), max_trials))
        try:
            tr, sq = EmitParsetree(cm, rng, name, prefix=prefix)
            return tr, sq, trial
        except PrefixMismatch:
            continue


# ---------------------------------------------------------------------------
# Exact prefix-conditional forward sampler (replaces rejection sampling)
#
# For a prefix pi (tuple of residue codes / None-wildcards, length P) we
# compute inside-style DP tables over the CM state graph:
#   pre[v][k]      = P(subtree(v) emits >= P-k residues, first P-k match pi[k:])
#   exact[v][k][L] = P(subtree(v) emits exactly L residues matching pi[k:k+L])
# then ancestral-sample the tree conditioned on the prefix (continuing
# unconditionally past it).  Cost O(P*M), independent of P(prefix): a 16-nt
# prefix that early-screened rejection needs ~1e8 trials for costs essentially
# nothing per sequence.  Not byte-constrained (no C reference exists for this
# path); deterministic per seed.  The DP is float64 -- no float32/Kahan needed.
# ---------------------------------------------------------------------------

class _PrefixTables:
    """DP tables + conditional sampler for one (CM, prefix) pair."""

    def __init__(self, cm, prefix_codes):
        self.cm = cm
        self.pi = prefix_codes
        self.P = len(prefix_codes)
        self.M = cm.M
        self.selfp = sreEXP2(float(cm.el_selfsc))
        if (cm.flags & CMH_LOCAL_END
                and abs(self.selfp - 1.0) < 0.01):
            cm_Fail("EL self transition probability %f is too high, would emit "
                    "long (too long) EL insertions." % self.selfp)
        self.childw = [None] * cm.M
        for v in range(cm.M):
            if cm.sttype[v] != B_st:
                self.childw[v] = self._child_weights(v)
        self._build()
        if self.pre[0][0] == 0.0:
            cm_Fail("prefix '%s' has probability 0 under this CM (impossible)"
                    % prefix_tostr(prefix_codes))

    # ---- emission-match factors --------------------------------------
    def _nullm(self, k):
        x = self.pi[k]
        return 1.0 if x is None else float(self.cm.null[x])

    def _emat(self, v, idx):
        x = self.pi[idx]
        return 1.0 if x is None else float(self.cm.e[v][x])

    def _elsum(self, v, k):
        """P(an MP emission has its 5' residue matching pi[k], 3' free)."""
        x = self.pi[k]
        if x is None:
            return 1.0
        e = self.cm.e[v]
        return sum(float(e[x * K + b]) for b in range(K))

    def _epair(self, v, k1, k2):
        """P(an MP pair matches pi[k1] (5'), pi[k2] (3'))."""
        x1 = self.pi[k1]; x2 = self.pi[k2]
        e = self.cm.e[v]
        if x1 is None and x2 is None:
            return 1.0
        if x1 is None:
            return sum(float(e[a * K + x2]) for a in range(K))
        if x2 is None:
            return sum(float(e[x1 * K + b]) for b in range(K))
        return float(e[x1 * K + x2])

    # ---- child transition weights (mirrors EmitParsetree's FChoose) ---
    def _child_weights(self, v):
        """(y, w) children of non-B state v; w a probability (sums to 1)."""
        cm = self.cm
        if v == 0 and cm.flags & CMH_LOCAL_BEGIN:
            if cm.flags & CM_EMIT_NO_LOCAL_BEGINS:
                wv = [(cm.cfirst[0] + i, float(cm.root_trans[i]))
                      for i in range(cm.cnum[0])]
            else:
                wv = [(y, float(cm.begin[y])) for y in range(cm.M)]
        elif cm.flags & CMH_LOCAL_END:
            wv = [(cm.cfirst[v] + i, float(cm.t[v][i]))
                  for i in range(cm.cnum[v])]
            if not (cm.flags & CM_EMIT_NO_LOCAL_ENDS):
                wv.append((cm.M, float(cm.end[v])))
        else:
            wv = [(cm.cfirst[v] + i, float(cm.t[v][i]))
                  for i in range(cm.cnum[v])]
        tot = sum(w for _, w in wv)
        if tot > 0.0:
            wv = [(y, w / tot) for y, w in wv]
        return wv

    # ---- DP ----------------------------------------------------------
    def _build(self):
        cm = self.cm; P = self.P; M = self.M
        pre = [[0.0] * (P + 1) for _ in range(M + 1)]
        exact = [[[0.0] * (P + 1) for _ in range(P + 1)]
                 for _ in range(M + 1)]
        # k from P down to 0; within k, exact[v][k][*] (L asc, v desc) then
        # pre[v][k] (v desc).  Children always have higher state indices, and
        # IL/EL self-loops only reference larger k, so this is acyclic.
        for k in range(P, -1, -1):
            r = P - k
            # EL (state M): closed forms
            for L in range(1, P - k + 1):
                f = 1.0
                for i in range(L):
                    f *= self._nullm(k + i)
                exact[M][k][L] = (self.selfp ** (L - 1)) * (1.0 - self.selfp) * f
            if r == 0:
                pre[M][k] = 1.0
            else:
                f = 1.0
                for i in range(r):
                    f *= self._nullm(k + i)
                pre[M][k] = f * (self.selfp ** (r - 1))
            for v in range(M - 1, -1, -1):
                st = cm.sttype[v]
                for L in range(0, P - k + 1):
                    exact[v][k][L] = self._exact_state(v, k, L, st, exact)
                pre[v][k] = self._pre_state(v, k, r, st, pre, exact)
        self.pre = pre
        self.exact = exact

    def _exact_state(self, v, k, L, st, exact):
        cm = self.cm
        if st == E_st:
            return 1.0 if L == 0 else 0.0
        if st == B_st:
            left = cm.cfirst[v]; right = cm.cnum[v]
            tot = 0.0
            for ell in range(L + 1):
                tot += exact[left][k][ell] * exact[right][k + ell][L - ell]
            return tot
        if st == MP_st:
            if L < 2:
                return 0.0
            s = 0.0
            for y, w in self.childw[v]:
                s += w * exact[y][k + 1][L - 2]
            return self._epair(v, k, k + L - 1) * s
        if st in (ML_st, IL_st):
            if L < 1:
                return 0.0
            s = 0.0
            for y, w in self.childw[v]:
                s += w * exact[y][k + 1][L - 1]
            return self._emat(v, k) * s
        if st in (MR_st, IR_st):
            if L < 1:
                return 0.0
            s = 0.0
            for y, w in self.childw[v]:
                s += w * exact[y][k][L - 1]
            return s * self._emat(v, k + L - 1)
        # e = 0 states (D, S, ROOT, ...)
        s = 0.0
        for y, w in self.childw[v]:
            s += w * exact[y][k][L]
        return s

    def _pre_state(self, v, k, r, st, pre, exact):
        if r == 0:
            return 1.0
        cm = self.cm
        if st == E_st:
            return 0.0
        if st == B_st:
            left = cm.cfirst[v]; right = cm.cnum[v]
            tot = pre[left][k]
            for ell in range(r):
                tot += exact[left][k][ell] * pre[right][k + ell]
            return tot
        if st == MP_st:
            if r == 1:
                return self._elsum(v, k)
            s = 0.0
            for y, w in self.childw[v]:
                if r == 2:
                    s += w * (exact[y][k + 1][0] * self._epair(v, k, k + 1)
                              + pre[y][k + 1] * self._elsum(v, k))
                else:
                    s += w * (exact[y][k + 1][r - 2]
                              * self._epair(v, k, k + r - 1)
                              + pre[y][k + 1] * self._elsum(v, k))
            return s
        if st in (ML_st, IL_st):
            em = self._emat(v, k)
            if r == 1:
                return em
            s = 0.0
            for y, w in self.childw[v]:
                s += w * pre[y][k + 1]
            return em * s
        if st in (MR_st, IR_st):
            # [child][r]: to reach >= r residues the child emits >= r (its own
            # first r match, then this r is beyond the prefix and free) OR the
            # child emits exactly r-1 and this r is the r-th residue.  Child
            # lengths < r-1 give a subtree < r residues -- cannot match.
            s = 0.0
            for y, w in self.childw[v]:
                s += w * (pre[y][k]
                          + exact[y][k][r - 1] * self._emat(v, k + r - 1))
            return s
        s = 0.0
        for y, w in self.childw[v]:
            s += w * pre[y][k]
        return s

    def prefix_prob(self):
        """Analytical P(seq[1:P] = prefix) for this CM."""
        return self.pre[0][0]

    def sample(self, rng, name):
        """Draw (parsetree, Seq) conditioned on the prefix, then continue
        unconditionally past it."""
        return _CondWalker(self, rng, name).sample_one()


class _CondWalker:
    """Ancestral-sampling walker over the DP tables: emits one sequence (+
    parsetree) conditioned on the prefix, continuing unconditionally after."""

    def __init__(self, T, rng, name):
        self.T = T
        self.cm = T.cm
        self.rng = rng
        self.name = name
        self.pi = T.pi
        self.P = T.P
        self.tr = Parsetree()
        self.gsq = []
        self.N = 0

    # ---- low-level helpers -------------------------------------------
    def _emit(self, x):
        self.gsq.append(x)
        self.N += 1

    def _pick(self, items):
        """items: list of (weight, payload); draw payload with prob ~ weight."""
        w = [float(a) for a, _ in items]
        tot = sum(w)
        if tot <= 0.0:
            raise RuntimeError("zero-weight branch in conditional sampler "
                               "(internal error)")
        return items[self.rng.DChoose(w, len(w))][1]

    def _pick_child(self, v):
        return self._pick([(w, y) for y, w in self.T.childw[v]])

    def _draw_res_cond(self, evec, idx):
        x = self.pi[idx]
        if x is None:
            return self.rng.FChoose(evec, K)
        if float(evec[x]) <= 0.0:
            raise RuntimeError("zero-probability constrained emission at "
                               "prefix position %d" % idx)
        return x

    def _draw_pair(self, v, idx1, idx2):
        """Draw an MP (l, r) jointly, constrained to match pi[idx1] (5') and
        pi[idx2] (3'); idx2=None leaves the 3' residue unconstrained."""
        e = self.cm.e[v]
        w = []
        for x in range(K * K):
            ok = True
            if idx1 is not None:
                c = self.pi[idx1]
                if c is not None and x // K != c:
                    ok = False
            if ok and idx2 is not None:
                c = self.pi[idx2]
                if c is not None and x % K != c:
                    ok = False
            w.append(float(e[x]) if ok else 0.0)
        z = self.rng.DChoose(w, K * K)
        return z // K, z % K

    # ---- EL state (geometric null run) -------------------------------
    def _el_pre(self, k, tpos):
        r = self.P - k
        for i in range(r):
            x = self.pi[k + i]
            self._emit(x if x is not None else self.rng.FChoose(self.cm.null, K))
        selfp = self.T.selfp
        y = self.rng.FChoose([selfp, 1.0 - selfp], 2)
        while y == 0:
            self._emit(self.rng.FChoose(self.cm.null, K))
            y = self.rng.FChoose([selfp, 1.0 - selfp], 2)

    def _el_exact(self, k, L, tpos):
        for i in range(L):
            x = self.pi[k + i]
            self._emit(x if x is not None else self.rng.FChoose(self.cm.null, K))

    def _el_uncond(self, tpos):
        selfp = self.T.selfp
        y = self.rng.FChoose([selfp, 1.0 - selfp], 2)
        while y == 0:
            self._emit(self.rng.FChoose(self.cm.null, K))
            y = self.rng.FChoose([selfp, 1.0 - selfp], 2)

    # ---- unconditional emission from state v --------------------------
    def emit_uncond(self, v, parent, whichway):
        cm = self.cm
        tpos = InsertTraceNode(self.tr, parent, whichway, self.N + 1, -1, v)
        st = cm.sttype[v]
        if st == EL_st:
            self._el_uncond(tpos)
        elif st == E_st:
            pass
        elif st == B_st:
            self.emit_uncond(cm.cfirst[v], tpos, TRACE_LEFT_CHILD)
            self.emit_uncond(cm.cnum[v], tpos, TRACE_RIGHT_CHILD)
        elif st == MP_st:
            x = self.rng.FChoose(cm.e[v], K * K)
            self._emit(x // K)
            y = self._pick_child(v)
            self.emit_uncond(y, tpos, TRACE_LEFT_CHILD)
            self._emit(x % K)
        elif st in (ML_st, IL_st):
            self._emit(self.rng.FChoose(cm.e[v], K))
            y = self._pick_child(v)
            self.emit_uncond(y, tpos, TRACE_LEFT_CHILD)
        elif st in (MR_st, IR_st):
            y = self._pick_child(v)
            self.emit_uncond(y, tpos, TRACE_LEFT_CHILD)
            self._emit(self.rng.FChoose(cm.e[v], K))
        else:   # e = 0 (D, S, ROOT, ...)
            y = self._pick_child(v)
            self.emit_uncond(y, tpos, TRACE_LEFT_CHILD)
        self.tr.emitr[tpos] = self.N

    # ---- sample subtree(v) conditioned on prefix at index k ----------
    def sample_pre(self, v, k, parent, whichway):
        r = self.P - k
        if r == 0:
            self.emit_uncond(v, parent, whichway)
            return
        cm = self.cm
        tpos = InsertTraceNode(self.tr, parent, whichway, self.N + 1, -1, v)
        st = cm.sttype[v]
        if st == EL_st:
            self._el_pre(k, tpos)
        elif st == E_st:
            raise RuntimeError("impossible: E state reached in sample_pre")
        elif st == B_st:
            left = cm.cfirst[v]; right = cm.cnum[v]
            items = [(self.T.pre[left][k], ("a", None))]
            for ell in range(r):
                items.append((self.T.exact[left][k][ell]
                              * self.T.pre[right][k + ell], ("b", ell)))
            kind, ell = self._pick(items)
            if kind == "a":
                self.sample_pre(left, k, tpos, TRACE_LEFT_CHILD)
                self.emit_uncond(right, tpos, TRACE_RIGHT_CHILD)
            else:
                self.sample_exact(left, k, ell, tpos, TRACE_LEFT_CHILD)
                self.sample_pre(right, k + ell, tpos, TRACE_RIGHT_CHILD)
        elif st == MP_st:
            if r == 1:
                l, rr = self._draw_pair(v, k, None)
                self._emit(l)
                y = self._pick_child(v)
                self.emit_uncond(y, tpos, TRACE_LEFT_CHILD)
                self._emit(rr)
            else:
                # Branch weights must equal the DP terms in _pre_state:
                # "exact" (child exactly r-2, pair at both ends) carries the
                # factor epair(v,k,k+r-1); "pre" (child >= r-1, r beyond the
                # prefix) carries the marginal elsum(v,k).  Omitting either
                # biases the exact/pre split by epair/elsum (up to ~0.8 on
                # RF02679 16-mer).  With them, weights sum to pre[v][k].
                items = []; kinds = []
                for y, w in self.T.childw[v]:
                    items.append(w * self.T.exact[y][k + 1][r - 2]
                                 * self.T._epair(v, k, k + r - 1))
                    kinds.append((y, "exact"))
                    items.append(w * self.T.pre[y][k + 1]
                                 * self.T._elsum(v, k))
                    kinds.append((y, "pre"))
                y, kind = self._pick(list(zip(items, kinds)))
                if kind == "exact":
                    l, rr = self._draw_pair(v, k, k + r - 1)
                    self._emit(l)
                    self.sample_exact(y, k + 1, r - 2, tpos, TRACE_LEFT_CHILD)
                    self._emit(rr)
                else:
                    l, rr = self._draw_pair(v, k, None)
                    self._emit(l)
                    self.sample_pre(y, k + 1, tpos, TRACE_LEFT_CHILD)
                    self._emit(rr)
        elif st in (ML_st, IL_st):
            self._emit(self._draw_res_cond(cm.e[v], k))
            if r == 1:
                y = self._pick_child(v)
                self.emit_uncond(y, tpos, TRACE_LEFT_CHILD)
            else:
                y = self._pick([(w * self.T.pre[yy][k + 1], yy)
                                for yy, w in self.T.childw[v]])
                self.sample_pre(y, k + 1, tpos, TRACE_LEFT_CHILD)
        elif st in (MR_st, IR_st):
            items = []; kinds = []
            for y, w in self.T.childw[v]:
                items.append(w * self.T.pre[y][k])
                kinds.append((y, -1))
                items.append(w * self.T.exact[y][k][r - 1]
                             * self.T._emat(v, k + r - 1))
                kinds.append((y, r - 1))
            y, s = self._pick(list(zip(items, kinds)))
            if s == -1:
                self.sample_pre(y, k, tpos, TRACE_LEFT_CHILD)
                self._emit(self.rng.FChoose(cm.e[v], K))
            else:
                self.sample_exact(y, k, s, tpos, TRACE_LEFT_CHILD)
                self._emit(self._draw_res_cond(cm.e[v], k + s))
        else:   # e = 0
            y = self._pick([(w * self.T.pre[yy][k], yy)
                            for yy, w in self.T.childw[v]])
            self.sample_pre(y, k, tpos, TRACE_LEFT_CHILD)
        self.tr.emitr[tpos] = self.N

    # ---- sample subtree(v) emitting exactly L residues = pi[k:k+L] ----
    def sample_exact(self, v, k, L, parent, whichway):
        cm = self.cm
        tpos = InsertTraceNode(self.tr, parent, whichway, self.N + 1, -1, v)
        st = cm.sttype[v]
        if st == EL_st:
            self._el_exact(k, L, tpos)
        elif st == E_st:
            if L != 0:
                raise RuntimeError("impossible: E state in sample_exact "
                                   "with L>0")
        elif st == B_st:
            left = cm.cfirst[v]; right = cm.cnum[v]
            ell = self._pick([(self.T.exact[left][k][el]
                               * self.T.exact[right][k + el][L - el], el)
                              for el in range(L + 1)])
            self.sample_exact(left, k, ell, tpos, TRACE_LEFT_CHILD)
            self.sample_exact(right, k + ell, L - ell, tpos, TRACE_RIGHT_CHILD)
        elif st == MP_st:
            if L < 2:
                raise RuntimeError("impossible: MP in sample_exact with L<2")
            l, rr = self._draw_pair(v, k, k + L - 1)
            self._emit(l)
            y = self._pick([(w * self.T.exact[yy][k + 1][L - 2], yy)
                            for yy, w in self.T.childw[v]])
            self.sample_exact(y, k + 1, L - 2, tpos, TRACE_LEFT_CHILD)
            self._emit(rr)
        elif st in (ML_st, IL_st):
            if L < 1:
                raise RuntimeError("impossible: ML/IL in sample_exact with L<1")
            self._emit(self._draw_res_cond(cm.e[v], k))
            y = self._pick([(w * self.T.exact[yy][k + 1][L - 1], yy)
                            for yy, w in self.T.childw[v]])
            self.sample_exact(y, k + 1, L - 1, tpos, TRACE_LEFT_CHILD)
        elif st in (MR_st, IR_st):
            if L < 1:
                raise RuntimeError("impossible: MR/IR in sample_exact with L<1")
            y = self._pick([(w * self.T.exact[yy][k][L - 1], yy)
                            for yy, w in self.T.childw[v]])
            self.sample_exact(y, k, L - 1, tpos, TRACE_LEFT_CHILD)
            self._emit(self._draw_res_cond(cm.e[v], k + L - 1))
        else:   # e = 0
            y = self._pick([(w * self.T.exact[yy][k][L], yy)
                            for yy, w in self.T.childw[v]])
            self.sample_exact(y, k, L, tpos, TRACE_LEFT_CHILD)
        self.tr.emitr[tpos] = self.N

    def sample_one(self):
        self.sample_pre(0, 0, -1, TRACE_LEFT_CHILD)
        return self.tr, Seq(self.name, [ESLDSQ_SENTINEL] + self.gsq)


# ---------------------------------------------------------------------------
# Alignment construction (cm_parsetree.c Parsetrees2Alignment)
# ---------------------------------------------------------------------------

def is_gap(c):
    """esl_abc_CIsGap() for the character sets that appear in our alignments
    ('-' and '.' are the only gap chars; '~' and letters are not)."""
    return c == '-' or c == '.'


def rightjustify(s, offset, n):
    """rightjustify(): pull non-gap chars right within s[offset:offset+n]."""
    if n <= 0:
        return
    end = offset + n
    npos = end - 1
    opos = end - 1
    while opos >= offset:
        if is_gap(s[opos]):
            opos -= 1
        else:
            s[npos] = s[opos]
            npos -= 1
            opos -= 1
    while npos >= offset:
        s[npos] = '.'
        npos -= 1


def leftjustify(s, offset, n):
    """leftjustify(): pull non-gap chars left within s[offset:offset+n]."""
    if n <= 0:
        return
    end = offset + n
    npos = offset
    opos = offset
    while opos < end:
        if is_gap(s[opos]):
            opos += 1
        else:
            s[npos] = s[opos]
            npos += 1
            opos += 1
    while npos < end:
        s[npos] = '.'
        npos += 1


class MSA:
    def __init__(self, nseq, alen):
        self.nseq = nseq
        self.alen = alen
        self.aseq = [['.'] * alen for _ in range(nseq)]
        self.sqname = [''] * nseq
        self.sqlen = [0] * nseq
        self.wgt = [1.0] * nseq
        self.name = None
        self.desc = None
        self.au = None
        self.ss_cons = ['.'] * alen
        self.rf = ['.'] * alen


def Parsetrees2Alignment(cm, abc_sym, sqA, trA, nseq):
    """Parsetrees2Alignment() with do_full=TRUE, do_matchonly=FALSE,
    allow_trunc=FALSE, no postcodes, no insert/EL files."""
    emap = cm.emap
    clen = emap.clen

    matuse = [0] * (clen + 1)
    maxil = [0] * (clen + 1)
    maxel = [0] * (clen + 1)
    maxir = [0] * (clen + 1)
    ilmap = [0] * (clen + 1)
    elmap = [0] * (clen + 1)
    irmap = [0] * (clen + 1)
    iluse = [0] * (clen + 1)
    eluse = [0] * (clen + 1)
    iruse = [0] * (clen + 1)
    ifirst = [-1] * (clen + 1)
    elfirst = [-1] * (clen + 1)

    for cpos in range(clen + 1):
        matuse[cpos] = 0 if (cpos == 0) else 1

    # ---- maximum insert lengths / used match columns ----
    for i in range(nseq):
        for cpos in range(clen + 1):
            iluse[cpos] = eluse[cpos] = iruse[cpos] = 0
        prvnd = 0
        tr = trA[i]
        for tpos in range(tr.n):
            v = tr.state[tpos]
            mode = tr.mode[tpos]
            if cm.sttype[v] == EL_st:
                nd = prvnd
            else:
                nd = cm.ndidx[v]
            st = cm.sttype[v]
            if st == MP_st:
                if ModeEmitsLeft(mode):  matuse[emap.lpos[nd]] = 1
                if ModeEmitsRight(mode): matuse[emap.rpos[nd]] = 1
            elif st == ML_st:
                if ModeEmitsLeft(mode):  matuse[emap.lpos[nd]] = 1
            elif st == MR_st:
                if ModeEmitsRight(mode): matuse[emap.rpos[nd]] = 1
            elif st == IL_st:
                if ModeEmitsLeft(mode):  iluse[emap.lpos[nd]] += 1
            elif st == IR_st:
                if ModeEmitsRight(mode): iruse[emap.rpos[nd] - 1] += 1
            elif st == EL_st:
                eluse[emap.epos[nd]] = tr.emitr[tpos] - tr.emitl[tpos] + 1
            prvnd = nd
        for cpos in range(clen + 1):
            if iluse[cpos] > maxil[cpos]: maxil[cpos] = iluse[cpos]
            if eluse[cpos] > maxel[cpos]: maxel[cpos] = eluse[cpos]
            if iruse[cpos] > maxir[cpos]: maxir[cpos] = iruse[cpos]

    # ---- total alignment length and cpos -> apos maps ----
    alen = 0
    matmap = [-1] * (clen + 1)
    for cpos in range(clen + 1):
        if matuse[cpos]:
            matmap[cpos] = alen
            alen += 1
        else:
            matmap[cpos] = -1
        elmap[cpos] = alen
        alen += maxel[cpos]
        ilmap[cpos] = alen
        alen += maxil[cpos]
        alen += maxir[cpos]
        irmap[cpos] = alen - 1

    msa = MSA(nseq, alen)
    msa.name = cm.name
    msa.desc = "Synthetic sequence alignment generated by cmemit"
    msa.au = "Infernal %s" % INFERNAL_VERSION

    # ---- per-sequence placement ----
    for i in range(nseq):
        tmp_aseq = ['.'] * alen
        for cpos in range(clen + 1):
            if matmap[cpos] != -1:
                tmp_aseq[matmap[cpos]] = '-'
        for cpos in range(clen + 1):
            iluse[cpos] = iruse[cpos] = eluse[cpos] = 0
            ifirst[cpos] = elfirst[cpos] = -1
        prvnd = 0
        tr = trA[i]
        sq = sqA[i]
        for tpos in range(tr.n):
            v = tr.state[tpos]
            mode = tr.mode[tpos]
            if cm.sttype[v] == EL_st:
                nd = prvnd
            else:
                nd = cm.ndidx[v]
            st = cm.sttype[v]
            if st == MP_st:
                if ModeEmitsLeft(mode):
                    cpos = emap.lpos[nd]
                    apos = matmap[cpos]
                    rpos = tr.emitl[tpos]
                    tmp_aseq[apos] = abc_sym[sq.codes[rpos]]
                if ModeEmitsRight(mode):
                    cpos = emap.rpos[nd]
                    apos = matmap[cpos]
                    rpos = tr.emitr[tpos]
                    tmp_aseq[apos] = abc_sym[sq.codes[rpos]]
            elif st == ML_st:
                if ModeEmitsLeft(mode):
                    cpos = emap.lpos[nd]
                    apos = matmap[cpos]
                    rpos = tr.emitl[tpos]
                    tmp_aseq[apos] = abc_sym[sq.codes[rpos]]
            elif st == MR_st:
                if ModeEmitsRight(mode):
                    cpos = emap.rpos[nd]
                    apos = matmap[cpos]
                    rpos = tr.emitr[tpos]
                    tmp_aseq[apos] = abc_sym[sq.codes[rpos]]
            elif st == IL_st:
                if ModeEmitsLeft(mode):
                    cpos = emap.lpos[nd]
                    apos = ilmap[cpos] + iluse[cpos]
                    rpos = tr.emitl[tpos]
                    if iluse[cpos] == 0:
                        ifirst[cpos] = rpos
                    iluse[cpos] += 1
                    tmp_aseq[apos] = abc_sym[sq.codes[rpos]].lower()
            elif st == EL_st:
                cpos = emap.epos[nd]
                apos = elmap[cpos]
                eluse[cpos] = tr.emitr[tpos] - tr.emitl[tpos] + 1
                elfirst[cpos] = tr.emitl[tpos]
                for rpos in range(tr.emitl[tpos], tr.emitr[tpos] + 1):
                    tmp_aseq[apos] = abc_sym[sq.codes[rpos]].lower()
                    apos += 1
            elif st == IR_st:
                if ModeEmitsRight(mode):
                    cpos = emap.rpos[nd] - 1
                    apos = irmap[cpos] - iruse[cpos]
                    rpos = tr.emitr[tpos]
                    ifirst[cpos] = rpos
                    iruse[cpos] += 1
                    tmp_aseq[apos] = abc_sym[sq.codes[rpos]].lower()
            elif st == D_st:
                if (cm.stid[v] == MATP_D or cm.stid[v] == MATL_D) and ModeEmitsLeft(mode):
                    cpos = emap.lpos[nd]
                    if matuse[cpos]:
                        tmp_aseq[matmap[cpos]] = '-'
                if (cm.stid[v] == MATP_D or cm.stid[v] == MATR_D) and ModeEmitsRight(mode):
                    cpos = emap.rpos[nd]
                    if matuse[cpos]:
                        tmp_aseq[matmap[cpos]] = '-'
            prvnd = nd

        msa.aseq[i] = tmp_aseq
        msa.sqname[i] = sq.name
        msa.sqlen[i] = sq.n

        # ---- rejustify (default: split inserts in half) ----
        if not (cm.align_opts & CM_ALIGN_FLUSHINSERTS):
            # 5' ELs flush right, then 5' ILs flush right
            rightjustify(msa.aseq[i], 0, maxel[0])
            rightjustify(msa.aseq[i], maxel[0], maxil[0])
            # internal splits
            for cpos in range(1, clen):
                if maxel[cpos] > 1:
                    apos = matmap[cpos] + 1
                    nins = 0
                    while apos < alen and msa.aseq[i][apos].islower():
                        nins += 1
                        apos += 1
                    nins //= 2
                    rightjustify(msa.aseq[i], matmap[cpos] + 1 + nins, maxel[cpos] - nins)
                if maxil[cpos] > 1:
                    apos = matmap[cpos] + 1 + maxel[cpos]
                    nins = 0
                    while apos < alen and msa.aseq[i][apos].islower():
                        nins += 1
                        apos += 1
                    nins //= 2
                    rightjustify(msa.aseq[i], matmap[cpos] + 1 + maxel[cpos] + nins,
                                 maxil[cpos] - nins)
                if maxir[cpos] > 1:
                    apos = matmap[cpos + 1] - 1
                    nins = 0
                    while apos >= 0 and msa.aseq[i][apos].islower():
                        nins += 1
                        apos -= 1
                    nins += 1
                    nins //= 2
                    leftjustify(msa.aseq[i],
                                matmap[cpos] + 1 + maxel[cpos] + maxil[cpos],
                                maxir[cpos] - nins)
            # 3' IRs flush left
            leftjustify(msa.aseq[i],
                        matmap[clen] + 1 + maxel[clen] + maxil[clen],
                        maxir[clen])

    # ---- ss_cons and rf ----
    cmcons = cm.cmcons
    for cpos in range(clen + 1):
        if matuse[cpos]:
            if (cmcons.ct[cpos - 1] != -1 and matuse[cmcons.ct[cpos - 1] + 1] == 0):
                msa.ss_cons[matmap[cpos]] = '.'
            else:
                msa.ss_cons[matmap[cpos]] = cmcons.cstr[cpos - 1]
            msa.rf[matmap[cpos]] = (cm.rf[cpos] if (cm.flags & CMH_RF)
                                    else cmcons.cseq[cpos - 1])
        if maxil[cpos] > 0:
            for apos in range(ilmap[cpos], ilmap[cpos] + maxil[cpos]):
                msa.ss_cons[apos] = '.'
                msa.rf[apos] = '.'
        if maxel[cpos] > 0:
            for apos in range(elmap[cpos], elmap[cpos] + maxel[cpos]):
                msa.ss_cons[apos] = '~'
                msa.rf[apos] = '~'
        if maxir[cpos] > 0:
            for apos in range(irmap[cpos], irmap[cpos] - maxir[cpos], -1):
                msa.ss_cons[apos] = '.'
                msa.rf[apos] = '.'

    return msa


# ---------------------------------------------------------------------------
# esl_wuss.c: nopseudo / wuss2ct / ct2wuss
# ---------------------------------------------------------------------------

def esl_wuss_nopseudo(ss):
    out = []
    for c in ss:
        out.append('.' if c.isalpha() else c)
    return out


def esl_wuss2ct(ss, n):
    """Returns ct[0..n] (1-based positions, ct[0]=0)."""
    pda = [None] * 27
    for i in range(1, 27):
        pda[i] = []
    pda[0] = []
    ct = [0] * (n + 1)
    for pos in range(1, n + 1):
        c = ss[pos - 1]
        if c in '<([{':
            pda[0].append(pos)
        elif c in '>)]}':
            if not pda[0]:
                raise RuntimeError("no closing bracket at position %d" % pos)
            pair = pda[0].pop()
            if ((ss[pair - 1] == '<' and c != '>') or
                    (ss[pair - 1] == '(' and c != ')') or
                    (ss[pair - 1] == '[' and c != ']') or
                    (ss[pair - 1] == '{' and c != '}')):
                raise RuntimeError("brackets don't match at position %d" % pos)
            ct[pos] = pair
            ct[pair] = pos
        elif c.isupper():
            i = ord(c) - ord('A') + 1
            if pda[i] is None:
                pda[i] = []
            pda[i].append(pos)
        elif c.islower():
            i = ord(c) - ord('a') + 1
            if pda[i] is None or not pda[i]:
                raise RuntimeError("no partner for lowercase at position %d" % pos)
            pair = pda[i].pop()
            ct[pos] = pair
            ct[pair] = pos
        elif c not in ':,_-.~':
            raise RuntimeError("bogus character %r at position %d" % (c, pos))
    for i in range(27):
        if pda[i] is not None and pda[i]:
            raise RuntimeError("unpaired residues remain")
    return ct


def esl_ct2wuss(ct, n):
    """Returns a string of WUSS characters for ct[1..n]."""
    npairs = 0
    for j in range(1, n + 1):
        if ct[j] > 0 and j < ct[j]:
            npairs += 1
    cct = list(ct)
    rb = [-1] * 26
    ss = [':'] * n
    pda = []
    auxpk = []
    auxss = []
    npairs_reached = 0
    xpk = 0
    for j in range(1, n + 1):
        if cct[j] == 0 or cct[j] > j:
            pda.append(j)
        else:
            found_partner = False
            nfaces = 0
            minface = -1
            while pda:
                i = pda.pop()
                if i < 0:
                    nfaces += 1
                    if i < minface:
                        minface = i
                elif cct[i] == j:
                    found_partner = True
                    npairs_reached += 1
                    if nfaces > 1 and minface > -4:
                        minface -= 1
                    if   minface == -1: ss[i - 1] = '<'; ss[j - 1] = '>'
                    elif minface == -2: ss[i - 1] = '('; ss[j - 1] = ')'
                    elif minface == -3: ss[i - 1] = '['; ss[j - 1] = ']'
                    elif minface == -4: ss[i - 1] = '{'; ss[j - 1] = '}'
                    else:
                        raise RuntimeError("no such face code %d" % minface)
                    pda.append(minface)
                    while auxss:
                        i2 = auxss.pop()
                        if   nfaces == 0: ss[i2 - 1] = '_'
                        elif nfaces == 1: ss[i2 - 1] = '-'
                        else:             ss[i2 - 1] = ','
                    break
                elif cct[i] == 0:
                    if ct[i] == 0:
                        auxss.append(i)
                else:
                    auxpk.append(i)
            if not found_partner:
                raise RuntimeError("Cannot find left partner (%d) of base %d. "
                                   "Likely a triplet" % (ct[j], j))
        if auxpk:
            leftbound = cct[j]
            rightbound = leftbound + 1
            xpk = -1
            while auxpk:
                i = auxpk.pop()
                k = rightbound - 1
                while k > leftbound:
                    if cct[k] == 0:
                        k -= 1
                        continue
                    elif cct[k] > rightbound:
                        k -= 1
                        continue
                    elif cct[k] == i:
                        break
                    else:
                        k = leftbound
                        break
                if k == leftbound:
                    xpk += 1
                    while i < rb[xpk]:
                        xpk += 1
                    leftbound = rightbound if rightbound < cct[i] else cct[j]
                    rightbound = cct[i]
                npairs_reached += 1
                if xpk + ord('a') <= ord('z'):
                    if cct[i] > rb[xpk]:
                        rb[xpk] = cct[i]
                    ss[i - 1] = chr(xpk + ord('A'))
                    ss[cct[i] - 1] = chr(xpk + ord('a'))
                    cct[i] = 0
                    cct[ct[i]] = 0
                else:
                    raise RuntimeError("Don't have enough letters to describe all "
                                       "different pseudoknots.")
    if npairs != npairs_reached:
        raise RuntimeError("found %d out of %d pairs." % (npairs_reached, npairs))
    return "".join(ss)


# ---------------------------------------------------------------------------
# truncate_msa (cmemit.c) + esl_msa_ColumnSubset
# ---------------------------------------------------------------------------

def truncate_msa(go, msa, rng):
    """truncate_msa() in cmemit.c (set_spos/set_epos always TRUE here)."""
    alen = msa.alen
    useme = [0] * alen
    clen = 0
    for apos in range(alen):
        if not is_gap(msa.rf[apos]):
            clen += 1

    a5p = go.a5p
    a3p = go.a3p
    rnd_spos = (a5p == 0)
    rnd_epos = (a3p == 0)

    if a3p > clen:
        cm_Fail("with --a3p <n> option, <n> must be <= consensus length of "
                "CM (%d).\n" % clen)
    if (rnd_spos and not rnd_epos) or (not rnd_spos and rnd_epos):
        cm_Fail("with --a5p <n1> and --a3p <n2>, either <n1> and <n2> must be "
                "0, or neither must be 0")
    if rnd_spos:
        spos = rng.Roll(clen) + 1
    else:
        spos = a5p
    if rnd_epos:
        epos = rng.Roll(clen) + 1
    else:
        epos = a3p
    if spos > epos:
        if rnd_spos and rnd_epos:
            spos, epos = epos, spos
        else:
            cm_Fail("with --a5p <n1> and --a3p <n2>, <n1> must be <= <n2>")

    msa.ss_cons = esl_wuss_nopseudo(msa.ss_cons)
    ct = esl_wuss2ct(msa.ss_cons, alen)

    cc = 0
    for apos in range(alen):
        if cc < (spos - 1) or cc > epos:
            useme[apos] = 0
            if ct[apos + 1] != 0:
                ct[ct[apos + 1]] = 0
            ct[apos + 1] = 0
        else:
            useme[apos] = 1
        if not is_gap(msa.rf[apos]):
            cc += 1
            if cc == epos + 1:
                useme[apos] = 0
                if ct[apos + 1] != 0:
                    ct[ct[apos + 1]] = 0
                ct[apos + 1] = 0

    msa.ss_cons = list(esl_ct2wuss(ct, alen))
    # ColumnSubset
    new_aseq = []
    for i in range(msa.nseq):
        new_aseq.append([])
    new_ss = []
    new_rf = []
    for opos in range(alen):
        if not useme[opos]:
            continue
        for i in range(msa.nseq):
            new_aseq[i].append(msa.aseq[i][opos])
        new_ss.append(msa.ss_cons[opos])
        new_rf.append(msa.rf[opos])
    msa.aseq = new_aseq
    msa.ss_cons = new_ss
    msa.rf = new_rf
    msa.alen = len(new_rf)


# ---------------------------------------------------------------------------
# Background sequence generators (stats.c / esl_randomseq.c)
# ---------------------------------------------------------------------------

def CreateGenomicHMM():
    """CreateGenomicHMM(): the 5-state generative HMM parameters (exact
    doubles from stats.c, each row DNorm'd)."""
    sA = [0.157377049180328, 0.39344262295082, 0.265573770491803,
          0.00327868852459016, 0.180327868852459]
    # tAA rows (5 states x 5), exact values from stats.c lines 605-638
    tAA = [
        [0.999483637183643, 0.000317942006440604, 0.000185401071732768,
         2.60394763669618e-07, 1.27593434198113e-05],
        [9.76333640771184e-05, 0.99980020511745, 9.191359010352e-05,
         7.94413051888677e-08, 1.01684870641751e-05],
        [1.3223694798182e-07, 0.000155642887774602, 0.999700615549769,
         9.15079680034191e-05, 5.21013575048369e-05],
        [0.994252873563218, 0.0014367816091954, 0.0014367816091954,
         0.0014367816091954, 0.0014367816091954],
        [8.32138798088677e-06, 2.16356087503056e-05, 6.42411152124459e-05,
         1.66427759617735e-07, 0.999905635460297],
    ]
    # eAA rows (5 states x 4), exact values from stats.c lines 645-673
    eAA = [
        [0.370906566523225, 0.129213995153577, 0.130511270043053, 0.369368168280145],
        [0.305194882571888, 0.194580936415687, 0.192343972160245, 0.307880208852179],
        [0.238484980800698, 0.261262845707113, 0.261810301531792, 0.238441871960397],
        [0.699280575539568, 0.00143884892086331, 0.00143884892086331, 0.297841726618705],
        [0.169064007664923, 0.331718611320207, 0.33045427183482, 0.16876310918005],
    ]
    DNorm(sA)
    for row in tAA:
        DNorm(row)
    for row in eAA:
        DNorm(row)
    return sA, tAA, eAA


def SampleGenomicSequenceFromHMM(rng, sA, tAA, eAA, L):
    """Sample a length-L digital sequence from the generative HMM.
    Returns 1-based list (codes[0]=sentinel)."""
    dsq = [ESLDSQ_SENTINEL] + [0] * L + [ESLDSQ_SENTINEL]
    si = rng.DChoose(sA, 5)
    for x in range(1, L + 1):
        dsq[x] = rng.DChoose(eAA[si], K)
        si = rng.DChoose(tAA[si], 5)
    return dsq


def esl_rsq_xIID(rng, fq, L):
    """esl_rsq_xIID: iid draws from double fq. Returns 1-based list."""
    dsq = [ESLDSQ_SENTINEL] + [0] * L + [ESLDSQ_SENTINEL]
    for x in range(1, L + 1):
        dsq[x] = rng.DChoose(fq, K)
    return dsq


# ---------------------------------------------------------------------------
# Writers (esl_sqio_Write FASTA / esl_msafile_stockholm_Write)
# ---------------------------------------------------------------------------

class OutStream:
    """Writes raw bytes (never newline-translates), so output is identical on
    Windows and POSIX. stdout is flushed on exit exactly like the C code's
    block-buffered stdout."""

    def __init__(self, path):
        if path is None:
            self.f = sys.stdout.buffer
            self.owns = False
        else:
            self.f = open(path, "wb")
            self.owns = True

    def write(self, s):
        self.f.write(s.encode("ascii"))

    def close(self):
        if self.owns:
            self.f.close()


def write_fasta(out, sq, abc_sym):
    """esl_sqascii_WriteFasta (digital, no acc/desc)."""
    out.write(">" + sq.name + "\n")
    n = sq.n
    codes = sq.codes
    for pos in range(0, n, 60):
        chunk = "".join(abc_sym[codes[pos + 1 + j]] for j in range(min(60, n - pos)))
        out.write(chunk + "\n")


def write_stockholm(out, msa):
    """esl_msafile_stockholm_Write with cpl=200, no GS section (no wgts)."""
    maxname = max(len(nm) for nm in msa.sqname)
    maxgf = 2
    maxgc = 7                      # SS_cons and RF are present
    margin = maxname + 1
    if maxgc + 6 > margin:
        margin = maxgc + 6
    out.write("# STOCKHOLM 1.0\n")
    if msa.name is not None:
        out.write("#=GF %-*s %s\n" % (maxgf, "ID", msa.name))
    if msa.desc is not None:
        out.write("#=GF %-*s %s\n" % (maxgf, "DE", msa.desc))
    if msa.au is not None:
        out.write("#=GF %-*s %s\n" % (maxgf, "AU", msa.au))
    out.write("\n")
    cpl = 200
    currpos = 0
    first = True
    while currpos < msa.alen:
        acpl = min(cpl, msa.alen - currpos)
        if not first:
            out.write("\n")
        first = False
        for i in range(msa.nseq):
            chunk = "".join(msa.aseq[i][currpos:currpos + acpl])
            out.write("%-*s %s\n" % (margin - 1, msa.sqname[i], chunk))
        chunk = "".join(msa.ss_cons[currpos:currpos + acpl])
        out.write("#=GC %-*s %s\n" % (margin - 6, "SS_cons", chunk))
        chunk = "".join(msa.rf[currpos:currpos + acpl])
        out.write("#=GC %-*s %s\n" % (margin - 6, "RF", chunk))
        currpos += cpl
    out.write("//\n")


# ---------------------------------------------------------------------------
# Driver (mirrors cmemit.c main / master / emit_*)
# ---------------------------------------------------------------------------

def cm_Fail(msg):
    """cm_Fail(): "\nError: <msg>\n" on stderr, then exit(1).  stdout is
    flushed by the interpreter at exit, matching the C block-buffered stdout."""
    sys.stderr.write("\nError: " + msg + "\n")
    sys.stderr.flush()
    sys.exit(1)


class Options:
    def __init__(self):
        self.h = False
        self.o = None
        self.N = 10
        self.outmode = 'u'
        self.e = None
        self.l = False
        self.u5p = False
        self.u3p = False
        self.a5p = None
        self.a3p = None
        self.seed = 0
        self.iid = False
        self.rna = True
        self.dna = False
        self.idx = 1
        self.outformat = "Stockholm"
        self._outformat_set = False
        self.tfile = None
        self.exp = None
        self.hmmonly = False
        self.nohmmonly = False
        self.prefix = None
        self.prefix_codes = None
        self.max_trials = None      # None = run until -N satisfied; --max-trials caps it
        self.sampler = "forward"    # 'forward' exact conditional / 'reject' early-screened rejection
        self.cmfile = None
        self._n_outmode = 0


def parse_args(argv):
    o = Options()
    i = 0
    n = len(argv)
    while i < n:
        a = argv[i]
        if a == "-h":
            o.h = True
        elif a == "-o":
            i += 1; o.o = argv[i]
        elif a == "-N":
            i += 1; o.N = int(argv[i])
            if o.N <= 0:
                cm_Fail("n>0 required for -N")
        elif a == "-u":
            o.outmode = 'u'; o._n_outmode += 1
        elif a == "-a":
            o.outmode = 'a'; o._n_outmode += 1
        elif a == "-c":
            o.outmode = 'c'; o._n_outmode += 1
        elif a == "-e":
            i += 1; o.e = int(argv[i])
            if o.e <= 0:
                cm_Fail("n>0 required for -e")
        elif a == "-l":
            o.l = True
        elif a == "--u5p":
            o.u5p = True
        elif a == "--u3p":
            o.u3p = True
        elif a == "--a5p":
            i += 1; o.a5p = int(argv[i])
            if o.a5p < 0:
                cm_Fail("n>=0 required for --a5p")
        elif a == "--a3p":
            i += 1; o.a3p = int(argv[i])
            if o.a3p < 0:
                cm_Fail("n>=0 required for --a3p")
        elif a == "--seed":
            i += 1; o.seed = int(argv[i])
            if o.seed < 0:
                cm_Fail("n>=0 required for --seed")
        elif a == "--iid":
            o.iid = True
        elif a == "--rna":
            o.rna = True; o.dna = False
        elif a == "--dna":
            o.dna = True; o.rna = False
        elif a == "--idx":
            i += 1; o.idx = int(argv[i])
            if o.idx <= 0:
                cm_Fail("n>0 required for --idx")
        elif a == "--outformat":
            i += 1; o.outformat = argv[i]; o._outformat_set = True
        elif a == "--tfile":
            i += 1; o.tfile = argv[i]
        elif a == "--exp":
            i += 1; o.exp = float(argv[i])
            if o.exp <= 0.0:
                cm_Fail("x>0 required for --exp")
        elif a == "--hmmonly":
            o.hmmonly = True
        elif a == "--nohmmonly":
            o.nohmmonly = True
        elif a == "--prefix":
            i += 1; o.prefix = argv[i]
            if len(o.prefix) == 0:
                cm_Fail("empty prefix given to --prefix")
        elif a == "--max-trials":
            i += 1; o.max_trials = int(argv[i])
            if o.max_trials <= 0:
                cm_Fail("n>0 required for --max-trials")
        elif a == "--sampler":
            i += 1; o.sampler = argv[i]
            if o.sampler not in ("forward", "reject"):
                cm_Fail("--sampler must be 'forward' or 'reject'")
        elif a.startswith("-") and a != "-":
            cm_Fail("Unrecognized option: %s" % a)
        else:
            o.cmfile = a
        i += 1

    # ---- constraint checks (esl_getopts groups) ----
    if o._n_outmode > 1:
        cm_Fail("mutually exclusive options: -u, -c, -a")
    if o.rna and o.dna:
        cm_Fail("mutually exclusive options: --rna, --dna")
    if o.hmmonly and o.nohmmonly:
        cm_Fail("mutually exclusive options: --hmmonly, --nohmmonly")
    if o.e is not None and o.outmode in ('a', 'c'):
        cm_Fail("options -e and -%s are mutually exclusive" % o.outmode)
    if (o.u5p or o.u3p) and o.outmode in ('a', 'c'):
        cm_Fail("options --u5p/--u3p and -%s are mutually exclusive" % o.outmode)
    if o.iid and o.e is None:
        cm_Fail("option --iid requires -e")
    if o._outformat_set and o.outmode != 'a':
        cm_Fail("option --outformat requires -a")
    if o.a5p is not None:
        if o.a3p is None or o.outmode != 'a':
            cm_Fail("option --a5p requires --a3p and -a")
    if o.a3p is not None:
        if o.a5p is None or o.outmode != 'a':
            cm_Fail("option --a3p requires --a5p and -a")
    if o.tfile is not None:
        if o.outmode == 'c' or o.e is not None or o.u5p or o.u3p:
            cm_Fail("option --tfile is incompatible with -c, -e, --u5p, --u3p")
    if o.prefix is not None and o.outmode == 'c':
        cm_Fail("option --prefix is incompatible with -c "
                "(the consensus sequence is deterministic)")
    if o.cmfile is None:
        cm_Fail("cmemit: no CM file given")
    return o


def CMCountNodetype(cm, ndtype):
    return sum(1 for t in cm.ndtype if t == ndtype)


def master(o, out):
    rng = Randomness(o.seed)
    cms = read_cm_file(o.cmfile)
    ncm = 0
    for cm in cms:
        ncm += 1
        if o.nohmmonly:
            use_cm = True
        elif o.hmmonly:
            use_cm = False
        else:
            use_cm = (CMCountNodetype(cm, MATP_nd) > 0)
        if not use_cm:
            raise NotImplementedError(
                "model %s has zero basepairs; emitting from the filter HMM is "
                "not implemented. Use --nohmmonly to force CM emission."
                % (cm.name if cm.name is not None else ncm))
        # initialize_cm
        if o.l:
            cm.config_opts |= CM_CONFIG_LOCAL
        if o.exp is not None:
            cm_Exponentiate(cm, o.exp)
        cm_Configure(cm)
        if o.prefix is not None and o.prefix_codes is None:
            abc_sym = DNA_SYM if o.dna else RNA_SYM
            o.prefix_codes = str2prefix(o.prefix, abc_sym)
        if o.outmode == 'c':
            emit_consensus(o, cm, out)
        elif o.outmode == 'a':
            emit_alignment(o, cm, rng, out, ncm)
        else:
            emit_unaligned(o, cm, rng, out, ncm)


def emit_consensus(o, cm, out):
    """emit_consensus(): write the consensus sequence as FASTA (text)."""
    if cm.cmcons is None:
        raise NotImplementedError("consensus sequence requires a configured CM "
                                  "(CM path). Use --nohmmonly.")
    if cm.name is None:
        raise NotImplementedError("consensus sequence requires a named CM")
    name = cm.name + "-cmconsensus"
    cseq = cm.cmcons.cseq
    out.write(">" + name + "\n")
    for pos in range(0, len(cseq), 60):
        out.write(cseq[pos:pos + 60] + "\n")


def _make_prefix_tables(o, cm):
    """Shared _PrefixTables for one CM when the forward conditional sampler is
    active (prefix set and o.sampler == 'forward'); None otherwise.  Constructed
    once per CM so the DP cost is paid once and reused across the N sequences.
    Raises cm_Fail immediately for an impossible prefix (pre[0][0] == 0)."""
    if o.prefix_codes is not None and o.sampler == "forward":
        return _PrefixTables(cm, o.prefix_codes)
    return None


def _emit_seq(o, cm, rng, i, ncm, embed_ctx, pt=None):
    """Emit one unaligned sequence #(i + idx): the prefix-screened (or plain)
    CM walk, then optional --u5p/--u3p truncation, then optional -e embedding.
    ``embed_ctx`` is the tuple from CreateGenomicHMM()/iid, or None without -e.
    ``pt`` is the shared _PrefixTables for the forward conditional sampler
    (None when not prefix-constrained or when using rejection sampling).

    Returns (sq2print, ntrials): the final Seq and the rejection-trial count
    (None without --prefix; 1 for the forward sampler).  Shared by
    emit_unaligned() (which writes it) and emit() (which returns it), so both
    produce identical draws and output.
    """
    offset = o.idx
    if cm.name is not None:
        name = "%s-sample%d" % (cm.name, i + offset)
    else:
        name = "%d-sample%d" % (ncm, i + offset)
    if pt is not None:
        tr, sq2print = pt.sample(rng, name)
        ntrials = 1
    elif o.prefix_codes is not None:
        tr, sq2print, ntrials = sample_one_with_prefix(cm, rng, o.prefix_codes,
                                                       name, o.max_trials)
    else:
        tr, sq2print = EmitParsetree(cm, rng, name)
        ntrials = None

    # truncate 5'/3'
    if o.u5p or o.u3p:
        start = rng.Roll(sq2print.n) + 1 if o.u5p else 1
        end = rng.Roll(sq2print.n) + 1 if o.u3p else sq2print.n
        if start > end:
            start, end = end, start
        sq2print = Seq(sq2print.name,
                       [ESLDSQ_SENTINEL] + sq2print.codes[start:end + 1])

    # embed
    if o.e is not None:
        embedL = o.e
        if sq2print.n > embedL:
            cm_Fail("<n>=%d from -eL <n> too small for emitted seq of "
                    "length %d, increase <n> and rerun" % (embedL, sq2print.n))
        fq, sA, tAA, eAA = embed_ctx
        if o.iid:
            bg = esl_rsq_xIID(rng, fq, embedL)
        else:
            bg = SampleGenomicSequenceFromHMM(rng, sA, tAA, eAA, embedL)
        if o.u5p:
            start = 1
        elif o.u3p:
            start = embedL - sq2print.n + 1
        else:
            start = rng.Roll(embedL - sq2print.n + 1) + 1
        for x in range(start, start + sq2print.n):
            bg[x] = sq2print.codes[x - start + 1]
        gname = "%s/%d-%d" % (sq2print.name, start, start + sq2print.n - 1)
        sq2print = Seq(gname, bg)
        sq2print.n = embedL          # bg has two sentinels; n = embedL, not len-1

    return sq2print, ntrials


def _gen_unaligned(o, cm, rng, ncm, pt=None):
    """Yield (final Seq, rejection-trials-or-None) for each of the N unaligned
    sequences of one CM, in emission order.  ``pt`` is the shared _PrefixTables
    for the forward conditional sampler (None otherwise).

    A generator (not a list-returning collector) so that the CLI writes each
    sequence the moment it is produced: on a mid-run cm_Fail (e.g. -e <n> too
    small), the output written so far is byte-identical to the C code, which
    also writes each sequence before moving on.
    """
    N = o.N
    if o.e is not None and o.u5p and o.u3p:
        cm_Fail("-e works in combination with --u5p or --u3p but not both")
    if o.e is not None:
        if o.iid:
            embed_ctx = ([1.0 / float(K)] * K, None, None, None)
        else:
            sA, tAA, eAA = CreateGenomicHMM()
            embed_ctx = (None, sA, tAA, eAA)
    else:
        embed_ctx = None
    for i in range(N):
        yield _emit_seq(o, cm, rng, i, ncm, embed_ctx, pt)


def emit_unaligned(o, cm, rng, out, ncm):
    abc_sym = DNA_SYM if o.dna else RNA_SYM
    pt = _make_prefix_tables(o, cm)
    for sq2print, _ in _gen_unaligned(o, cm, rng, ncm, pt=pt):
        write_fasta(out, sq2print, abc_sym)


def _collect_alignment(o, cm, rng, ncm, abc_sym, pt=None):
    """Emit all N sequences + parsetrees for one CM and build their alignment.
    Returns (msa, trials): the MSA and a list of per-seq rejection-trial counts
    (None entries without --prefix; 1s for the forward sampler).  Mirrors
    emit_alignment()'s draw order, including the optional --a5p/--a3p
    truncation of the MSA."""
    N = o.N
    offset = o.idx
    trA = []
    sqA = []
    trials = []
    for i in range(N):
        if cm.name is not None:
            name = "%s-sample%d" % (cm.name, i + offset)
        else:
            name = "%d-sample%d" % (ncm, i + offset)
        if pt is not None:
            tr, sq = pt.sample(rng, name)
            ntrials = 1
        elif o.prefix_codes is not None:
            tr, sq, ntrials = sample_one_with_prefix(cm, rng, o.prefix_codes,
                                                     name, o.max_trials)
        else:
            tr, sq = EmitParsetree(cm, rng, name)
            ntrials = None
        trA.append(tr)
        sqA.append(sq)
        trials.append(ntrials)

    msa = Parsetrees2Alignment(cm, abc_sym, sqA, trA, N)
    if o.a5p is not None and o.a3p is not None:
        truncate_msa(o, msa, rng)
    return msa, trials


def emit_alignment(o, cm, rng, out, ncm):
    abc_sym = DNA_SYM if o.dna else RNA_SYM
    outfmt = o.outformat
    if outfmt.lower() != "stockholm":
        raise NotImplementedError("--outformat %s: only 'Stockholm' is "
                                  "implemented" % outfmt)
    pt = _make_prefix_tables(o, cm)
    msa, _ = _collect_alignment(o, cm, rng, ncm, abc_sym, pt=pt)
    write_stockholm(out, msa)


def usage(out):
    out.write("Usage: pycmemit [-options] <cmfile>\n\n"
              "Options:\n"
              "  -h         : help\n"
              "  -o <f>     : send output to file <f>, not stdout\n"
              "  -N <n>     : generate <n> sequences [10]\n"
              "  -u/-a/-c   : unaligned FASTA / alignment / consensus [u]\n"
              "  -e <n>     : embed in random seq of length <n>\n"
              "  -l         : local configuration\n"
              "  --u5p/--u3p: truncate unaligned seqs 5'/3'\n"
              "  --a5p/--a3p <n>: truncate alignment at match col <n> (0=random)\n"
              "  --seed <n> : RNG seed (0 not allowed; use an explicit seed)\n"
              "  --iid      : with -e, iid background\n"
              "  --rna/--dna: output alphabet [rna]\n"
              "  --idx <n>  : start sequence numbering at <n> [1]\n"
              "  --exp <x>  : exponentiate CM probs by <x>\n"
              "  --nohmmonly: always emit from the CM\n"
              "  --hmmonly  : emit from filter HMM (NOT IMPLEMENTED)\n"
              "  --prefix <s>: each emitted seq's 5' end starts with <s>\n"
              "                (IUPAC ambiguity codes match any residue)\n"
              "  --sampler <f|r>: prefix sampler; forward (exact DP, default)\n"
              "                or reject (early-screened rejection sampling)\n"
              "  --max-trials <n>: rejection trials per seq before giving up\n"
              "                (reject sampler only) [unlimited if not given]\n")


def main(argv):
    o = parse_args(argv[1:])
    if o.h:
        usage(sys.stdout)
        return 0
    out = OutStream(o.o)
    master(o, out)
    out.close()
    return 0


# ---------------------------------------------------------------------------
# Importable API: emit()  (wraps the same machinery; returns a dict)
# ---------------------------------------------------------------------------

def _seq2str(sq, abc_sym):
    """ASCII string of a digital Seq, using sq.n (not len-1) so embedded
    sequences (two sentinels) serialize exactly as write_fasta does."""
    return "".join(abc_sym[c] for c in sq.codes[1:sq.n + 1])


def _validate_options(o):
    """Option-group validation for emit(), mirroring parse_args() but raising
    ValueError instead of cm_Fail (which would exit the calling process)."""
    if o.outmode not in ('u', 'a', 'c'):
        raise ValueError("outmode must be 'u', 'a' or 'c', got %r" % o.outmode)
    if o.N <= 0:
        raise ValueError("n>0 required for -N")
    if o.seed <= 0:
        raise ValueError("--seed 0 (one-time arbitrary seed) is not supported; "
                         "give an explicit seed > 0")
    if o.e is not None:
        if o.e <= 0:
            raise ValueError("n>0 required for -e")
        if o.outmode in ('a', 'c'):
            raise ValueError("options -e and -%s are mutually exclusive"
                             % o.outmode)
    if (o.u5p or o.u3p) and o.outmode in ('a', 'c'):
        raise ValueError("options --u5p/--u3p and -%s are mutually exclusive"
                         % o.outmode)
    if o.iid and o.e is None:
        raise ValueError("option --iid requires -e")
    if o.exp is not None and o.exp <= 0.0:
        raise ValueError("x>0 required for --exp")
    if o.a5p is not None:
        if o.a3p is None or o.outmode != 'a':
            raise ValueError("option --a5p requires --a3p and -a")
    if o.a3p is not None:
        if o.a5p is None or o.outmode != 'a':
            raise ValueError("option --a3p requires --a5p and -a")
    if o.idx <= 0:
        raise ValueError("n>0 required for --idx")
    if o.hmmonly and o.nohmmonly:
        raise ValueError("mutually exclusive options: --hmmonly, --nohmmonly")
    if o.prefix is not None:
        if len(o.prefix) == 0:
            raise ValueError("empty prefix given to --prefix")
        if o.outmode == 'c':
            raise ValueError("option --prefix is incompatible with -c "
                             "(the consensus sequence is deterministic)")
    if o.max_trials is not None and o.max_trials <= 0:
        raise ValueError("n>0 required for --max-trials")
    if o.sampler not in ("forward", "reject"):
        raise ValueError("sampler must be 'forward' or 'reject', got %r"
                         % o.sampler)
    if o.cmfile is None:
        raise ValueError("cmemit: no CM file given")
    if not os.path.isfile(o.cmfile):
        raise ValueError("cmfile not found: %r" % o.cmfile)


def _model_to_dict(o, cm, rng, ncm):
    """Generate the results for one CM and pack them into the per-model dict."""
    abc_sym = DNA_SYM if o.dna else RNA_SYM
    pt = _make_prefix_tables(o, cm)
    m = {"model": cm.name if cm.name is not None else str(ncm)}
    if o.outmode == 'c':
        if cm.cmcons is None:
            raise NotImplementedError("consensus sequence requires a configured "
                                      "CM (CM path). Use nohmmonly=True.")
        if cm.name is None:
            raise NotImplementedError("consensus sequence requires a named CM")
        cseq = cm.cmcons.cseq
        m["names"] = [cm.name + "-cmconsensus"]
        m["sequences"] = [cseq]
        m["lengths"] = [len(cseq)]
        m["trials"] = None
        m["acceptance"] = None
    elif o.outmode == 'a':
        msa, trials = _collect_alignment(o, cm, rng, ncm, abc_sym, pt=pt)
        m["names"] = list(msa.sqname)
        m["sequences"] = ["".join(row) for row in msa.aseq]
        m["lengths"] = list(msa.sqlen)
        m["trials"] = None
        m["acceptance"] = None
        m["ss_cons"] = "".join(msa.ss_cons)
        m["rf"] = "".join(msa.rf)
        buf = StringIO()
        write_stockholm(buf, msa)
        m["stockholm"] = buf.getvalue()
    else:   # 'u'
        seqs, trials = [], []
        for sq2print, ntrials in _gen_unaligned(o, cm, rng, ncm, pt=pt):
            seqs.append(sq2print)
            trials.append(ntrials)
        m["names"] = [sq.name for sq in seqs]
        m["sequences"] = [_seq2str(sq, abc_sym) for sq in seqs]
        m["lengths"] = [sq.n for sq in seqs]
        m["trials"] = None
        m["acceptance"] = None
    if o.prefix_codes is not None:
        ts = [t for t in trials if t is not None]
        if ts:
            total = sum(ts)
            m["trials"] = total
            m["acceptance"] = o.N / float(total)
        m["prefix_prob"] = (pt.prefix_prob() if pt is not None else None)
    return m


def _emit_to_dict(o):
    """The emit() core: configure each CM like master() but collect results."""
    rng = Randomness(o.seed)
    cms = read_cm_file(o.cmfile)
    if not cms:
        raise ValueError("no CMs found in %s" % o.cmfile)
    result = {
        "ok": True,
        "mode": o.outmode,
        "alphabet": "DNA" if o.dna else "RNA",
        "seed": o.seed,
        "N": o.N,
        "prefix": o.prefix,
        "cmfile": o.cmfile,
        "nmodels": 0,
        "models": [],
    }
    ncm = 0
    for cm in cms:
        ncm += 1
        if o.nohmmonly:
            use_cm = True
        elif o.hmmonly:
            use_cm = False
        else:
            use_cm = (CMCountNodetype(cm, MATP_nd) > 0)
        if not use_cm:
            raise NotImplementedError(
                "model %s has zero basepairs; emitting from the filter HMM is "
                "not implemented. Use nohmmonly=True."
                % (cm.name if cm.name is not None else ncm))
        # initialize_cm
        if o.l:
            cm.config_opts |= CM_CONFIG_LOCAL
        if o.exp is not None:
            cm_Exponentiate(cm, o.exp)
        cm_Configure(cm)
        if o.prefix is not None and o.prefix_codes is None:
            abc_sym = DNA_SYM if o.dna else RNA_SYM
            o.prefix_codes = str2prefix(o.prefix, abc_sym)
        result["models"].append(_model_to_dict(o, cm, rng, ncm))
    result["nmodels"] = len(result["models"])
    return result


def emit(cmf, N=10, outmode="u", seed=42, prefix=None, max_trials=None,
         sampler="forward", dna=False, local=False, exp=None, embed=None,
         iid=False, u5p=False, u3p=False, a5p=None, a3p=None, idx=1,
         nohmmonly=True, hmmonly=False):
    """Generate sequences from an Infernal covariance-model file.

    This is the importable entry point for pycmemit::

        from pycmemit import emit
        res = emit("4.c.cm", N=50, prefix="GG", seed=42)
        for s in res["models"][0]["sequences"]:
            print(s)

    Parameters mirror the cmemit CLI options:

      cmf         path to the INFERNAL1/a .cm file (one or more CMs)
      N           number of sequences to emit per CM [10]
      outmode     'u' unaligned FASTA-style / 'a' alignment / 'c' consensus [u]
      seed        RNG seed; must be > 0 (seed 0 is not reproducible) [42]
      prefix      5' sequence every emitted sequence must start with
                  (None = no constraint; IUPAC ambiguity codes are wildcards)
      sampler     prefix sampler, 'forward' (default): exact DP conditional
                  sampler, O(prefix*states) per CM, works for arbitrarily rare
                  prefixes; 'reject': early-screened rejection sampling (the
                  old behavior), costs ~1/P(prefix) trials per sequence
      max_trials  rejection-sampling cap per sequence (reject sampler only;
                  ignored by 'forward') [None = run until N are produced]
      dna         emit DNA alphabet instead of RNA
      local       local configuration (-l)
      exp         exponentiate CM probabilities by <x>
      embed       embed each sequence in a random background of length <n> (-e)
      iid         with embed, use an iid background instead of the genomic HMM
      u5p/u3p     truncate unaligned sequences at the 5'/3' end
      a5p/a3p     truncate the alignment at match column <n> (0 = random)
      idx         start sequence numbering at <idx> [1]
      nohmmonly   force CM emission (the only implemented path) [True]
      hmmonly     emit from the filter HMM (not implemented; raises)

    Returns a dict:

      {"ok", "mode", "alphabet", "seed", "N", "prefix", "cmfile", "nmodels",
       "models": [{"model", "names", "sequences", "lengths",
                   "trials", "acceptance", "prefix_prob",
                   "ss_cons", "rf", "stockholm"}, ...]}

    per-model fields:

      names       sequence names (e.g. "tRNA-sample1")
      sequences   mode 'u': the emitted strings (post embed/truncate if asked)
                  mode 'a': aligned rows (all length = alignment width)
                  mode 'c': [consensus string]
      lengths     sequence lengths (unaligned) for 'u'; the emitted lengths
                  (msa.sqlen) for 'a'; consensus length for 'c'
      trials      rejection-sampling trials across the N sequences
                  (only when prefix is set, else None)
      acceptance  N / trials (only when prefix is set, else None)
      prefix_prob analytical P(prefix) under this CM (forward sampler, only
                  when prefix is set; None for 'reject')
      ss_cons/rf  consensus secondary structure / reference annotation ('a')
      stockholm   full Stockholm 1.0 block text ('a')

    Errors are raised as exceptions (ValueError for bad options, RuntimeError
    for cm_Fail conditions such as an impossible prefix), never as
    ``sys.exit``.  With the default 'forward' sampler an impossible prefix
    (probability exactly 0) fails immediately instead of timing out.
    """
    o = Options()
    o.N = N
    o.outmode = outmode
    o.seed = seed
    o.prefix = prefix
    o.max_trials = max_trials
    o.sampler = sampler
    o.dna = dna
    o.rna = not dna
    o.l = local
    o.exp = exp
    o.e = embed
    o.iid = iid
    o.u5p = u5p
    o.u3p = u3p
    o.a5p = a5p
    o.a3p = a3p
    o.idx = idx
    o.nohmmonly = nohmmonly
    o.hmmonly = hmmonly
    o.cmfile = cmf

    _validate_options(o)
    try:
        return _emit_to_dict(o)
    except SystemExit:                    # cm_Fail() inside the machinery
        raise RuntimeError("pycmemit failed (see error above)") from None


__all__ = ["emit"]


if __name__ == "__main__":
    sys.exit(main(sys.argv))
