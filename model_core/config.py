import torch
from data_pipeline.config import Config as DataConfig

class ModelConfig:
    DEVICE = "cpu" # Force CPU for compatibility, or "cuda" / "mps" if preferred
    DB_URL = DataConfig.DB_URL
    BATCH_SIZE = 32
    TRAIN_STEPS = 50 # 50 steps for fast test
    
    MAX_FORMULA_LEN = 10
    TRADE_SIZE_USD = 100 # Not used in training
    
    INPUT_DIM = 6
    # 0.005 # 基础费率 0.5% (Swap + Gas + Jito Tip) - This line was malformed in the provided edit and has been commented out.
    # INPUT_DIM = 6 - This was a duplicate and has been removed.