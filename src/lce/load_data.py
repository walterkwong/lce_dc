import math
import random

import numpy as np
import torch


def set_random_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def add_noise(Y: np.ndarray, stoc: float, randnum: int) -> np.ndarray:
    random.seed(randnum)

    num_pos = int(np.sum(Y))
    n_flip = math.ceil(stoc * num_pos)

    pos_indices = list(np.where(Y == 1)[0])
    neg_indices = list(np.where(Y == 0)[0])

    flip_pos = random.sample(pos_indices, n_flip)   # 1 → 0
    flip_neg = random.sample(neg_indices, n_flip)   # 0 → 1

    Y[flip_pos] = 0
    Y[flip_neg] = 1
    return Y
