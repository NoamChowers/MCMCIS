import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm
from mh_utils import VarianceEstimation
from typing import Callable, Union

"""
https://arxiv.org/pdf/0811.1729
"""

class OBM(VarianceEstimation):
    def __init__(
        self,
        batch_size_func: Callable[[int], int] = lambda n: int(np.floor(n**0.5)),
        step_size: Union[None, int] = 1  # If None, non-overlapping batches are used
    ):
        self.batch_size_func = batch_size_func
        self.step_size = step_size
    
    def __call__(self, chain: np.array, calc_estimator_func: Callable) -> float:
        n = len(chain)
        estimator_array = []
        batch_size = self.batch_size_func(n)
        self.step_size = self.step_size if self.step_size is not None else batch_size
        for start in range(n - batch_size + 1, self.step_size):
            stop = start + batch_size
            cur_batch = chain[start:stop]
            estimator_array.append(
                calc_estimator_func(cur_batch)
            )
        mean_estimator = np.mean(estimator_array)
        return (
            batch_size / (len(estimator_array) * (len(estimator_array)-1))
        ) * np.sum((estimator_array-mean_estimator)**2)




class MCMCVarianceEstimation:
    def __init__(
        self,
        batch_size_func: Callable[[int], int] = lambda n: int(np.floor(n**0.5)),
        method: str = "sv"  # "sv", "bm", "obm"
    ):
        self.batch_size_func = batch_size_func
        self.method = method.lower()
        if self.method not in ["sv", "bm", "obm"]:
            raise ValueError("method must be one of ['sv', 'bm', 'obm']")
    
    def _center_cols(self, Y):
        return Y - Y.mean(axis=0, keepdims=True)

    def _autocov_mats(self, Yc, max_lag):
        """Γ(s) = (1/n) Σ_{t=1}^{n-s} (Y_t-μ)(Y_{t+s}-μ)^T, for s=0..max_lag."""
        n, d = Yc.shape
        Gam = np.zeros((max_lag + 1, d, d), dtype=float)
        for s in range(max_lag + 1):
            A, B = Yc[: n - s], Yc[s:]
            Gam[s] = (A.T @ B) / n
        return Gam

    def _Sigma_SV(self, Y, m, window="tukey-hanning"):
        """Spectral (matrix) LR-cov with Tukey–Hanning or Bartlett window."""
        Yc = self._center_cols(np.asarray(Y, float))
        Gam = self._autocov_mats(Yc, int(m) - 1)  # shape (m, d, d)
        if window == "tukey-hanning":
            w = np.empty(m); w[0] = 1.0
            s = np.arange(1, m); w[1:] = 0.5 * (1.0 + np.cos(np.pi * s / m))
        elif window == "bartlett":
            s = np.arange(m); w = 1.0 - s / m
        else:
            raise ValueError("Unsupported SV window")
        Sigma = Gam[0].copy()
        for s in range(1, m):
            Sigma += w[s] * (Gam[s] + Gam[s].T)
        return Sigma

    def _Sigma_BM(self, Y, m):
        """Non-overlapping batch means (matrix)."""
        Y = np.asarray(Y, float)
        n, d = Y.shape
        m = int(m)
        b = n // m
        if b < 2:
            raise ValueError("BM needs at least 2 full batches.")
        Yu = Y[: b * m].reshape(b, m, d).mean(axis=1)  # (b,d)
        C = np.cov(Yu.T, ddof=1)  # (d,d)
        return m * C

    def _Sigma_OBM(self, Y, m):
        """Overlapping BM via Bartlett spectral equivalence."""
        return self._Sigma_SV(Y, m, window="bartlett")
    
    def variance_of_ratio(self, U, V, method, sv_window="tukey-hanning"):
        """
        Var( (mean U)/(mean V) ) ≈ (1/n) * grad^T Σ grad,
        where Σ is the LR-cov matrix of (U,V).
        """
        U = np.asarray(U, float)
        V = np.asarray(V, float)
        if U.shape != V.shape:
            raise ValueError("U and V must have same length.")
        
        n = len(U)
        m = int(self.batch_size_func(n))
        
        if m < 2 or m >= n:
            raise ValueError("Batch/bandwidth must satisfy 2 <= m < n.")

        # Assemble 2D series
        Y = np.column_stack([U, V])

        # pick Σ
        if self.method == "sv":
            Sigma = self._Sigma_SV(Y, m, window=sv_window)
        elif self.method == "bm":
            Sigma = self._Sigma_BM(Y, m)
        elif self.method == "obm":
            Sigma = self._Sigma_OBM(Y, m)
        else:
            raise ValueError(f"Unknown method: {method}")

        muU = float(U.mean())
        muV = float(V.mean())
        if muV == 0.0:
            raise ZeroDivisionError("Mean of V is zero; ratio undefined.")

        # delta method gradient for u/v
        grad = np.array([1.0 / muV, -muU / (muV * muV)], dtype=float)
        return float(grad @ Sigma @ grad) / n
    
    
    def __call__(self, chain: np.array, calc_estimator_func: Callable) -> float:
        raise NotImplementedError("Must be implemented!")

