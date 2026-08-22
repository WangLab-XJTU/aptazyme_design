"""Central config for the FC- pistol scorer (GCN + Conv1d on BPP + base identity)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
OUT_DIR = ROOT / "out"

RAW_CSV = DATA_DIR / "RawData.csv"
PISTOL_STO = DATA_DIR / "pistol.sto.txt"

# ribozyme domain in the 140-nt construct: 1-idx 71-130 (60 nt), 0-idx 70:129
CORE = slice(70, 130)
BASE = {"A": 0, "C": 1, "G": 2, "U": 3}
BPP_THRESHOLD = 1e-6

# model / training
NODE_DIM = 5                 # base one-hot (4) + BPP diagonal unpaired prob (1)
HIDDEN = 64
GNN_LAYERS = 2
CNN_CHANNELS = 64
CNN_KERNEL = 3
LR = 1e-3
BATCH_SIZE = 256
EPOCHS = 60
PATIENCE = 8
TRAIN_RATIO = 0.8
SEED = 0
MC_PASSES = 20
N_JOBS = -1                  # parallel workers for BPP etc. (-1 = all cores)

OUT_MODEL = OUT_DIR / "fcminus_gcn.pt"
