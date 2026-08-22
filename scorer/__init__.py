"""FC- pistol scorer: GCN + Conv1d on ribozyme-domain BPP + base identity.

Pipeline: pycmemit generates candidate cores -> this scorer ranks them by
predicted unbound activity (FC-) with an activity prior and confidence.
"""
from .config import (BATCH_SIZE, CACHE_DIR, CORE, DATA_DIR, EPOCHS, GNN_LAYERS,
                     HIDDEN, MC_PASSES, NODE_DIM, OUT_DIR, RAW_CSV, SEED)
