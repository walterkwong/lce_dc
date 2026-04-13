# LCP dataset names
ALL_DATA_NAMES: list[str] = [
    "Repeats1", "Repeats2", "Repeats3",
    "Order1",   "Order2",   "Order3",
    "Timing1",  "Timing2",  "Timing3", "Timing4",
    "Period1",  "Period2",  "Period3", "Period4",
]

# LCPs that require a higher starting data-size due to extreme sparsity
SPARSE_LCPS: set[str] = {"Period2", "Period3", "Period4"}

# Supported model names
MODEL_NAMES: list[str] = [
    "LR",
    "XGBoost",
    "CNN",
    "ResNet",
    "InceptionTime",
    "MLSTMFCN",
    "LSTMAttention",
]

# Evaluation metric column names
METRIC_COLS: list[str] = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "av_prec",
    "brier",
    "pr_auc",
]
