"""
Created by Constantin Philippenko, 8th November 2022.
"""
import random

import numpy as np


def diamond_distribution(size: int) -> np.ndarray:
    """Sample points from the diamond distribution."""
    return np.array([random.choice([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]) for k in range(size)])