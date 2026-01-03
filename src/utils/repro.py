# src/utils/repro.py
import os
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int = 0, deterministic: bool = True) -> None:
    """
    Seed Python / NumPy / PyTorch for reproducibility.
    deterministic=True: prefer deterministic behavior on CUDA (may be slightly slower).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # If you want extra-strict determinism (may raise errors for some ops):
        # torch.use_deterministic_algorithms(True)


def make_torch_generator(seed: int, device: Optional[str] = "cpu") -> torch.Generator:
    """
    A torch.Generator to make DataLoader(shuffle=True) reproducible.
    Use: DataLoader(..., shuffle=True, generator=make_torch_generator(seed))
    """
    g = torch.Generator(device=device if device is not None else "cpu")
    g.manual_seed(seed)
    return g
