import numpy as np


class RunningMeanStd:
    def __init__(self, dim: int):
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)

    def _update(self, obs: np.ndarray):
        self.count += 1
        delta = obs - self.mean
        self.mean += delta / self.count
        delta2 = obs - self.mean
        self.m2 += delta * delta2

    def normalize(self, obs: np.ndarray, update=True) -> np.ndarray:
        if update:
            self._update(obs)
        return (obs - self.mean) / (np.sqrt(self.m2 / self.count) + 1e-8)
