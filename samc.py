from typing import Callable, Union
import numpy as np

from mh_utils import AdaptiveImportanceSampling



class SAMC(AdaptiveImportanceSampling):
    def __init__(
        self,
        lambda_star: float,
        target_pdf: Callable = lambda x: 1,
        lower_bound: float = 0,
        num_regions: int = 100,
        t0: int = 10_000,
        region_probs: Union[None, np.array] = None,
    ):
        self.lower_bound = lower_bound
        self.lambda_star = lambda_star
        self.num_regions = num_regions
        self.t0 = t0
        self.theta = np.zeros(self.num_regions)
        self.theta_visits = np.zeros(self.num_regions)

        
        self.region_probs = (
            region_probs if region_probs is not None else np.array([1/self.num_regions] * self.num_regions)
        )  # Uniform if nothing set beforehand
        assert len(self.region_probs) == self.num_regions, "region_probs and num_regions must be of the same length."
        
        super().__init__(self.importance_pdf, self.log_importance_pdf, target_pdf, update_after_each_sample=True)
        self.initialize_parameters()

    def _calc_regions(self):
        self.lower_bounds = np.array([
            self.lower_bound + x * (self.lambda_star - self.lower_bound) / (self.num_regions - 1)
            for x in range(self.num_regions)
        ])

    def initialize_parameters(self):
        self._calc_regions()
        super().initialize_parameters()
    
    
    def _find_region(self, x):
        return max(
            np.searchsorted(self.lower_bounds, x, side="right") - 1,
            0
        )
    
    def importance_pdf(self, x):    
        return np.exp(-self.theta[self._find_region(x)])
    
    
    def log_importance_pdf(self, x):    
        return -self.theta[self._find_region(x)]

    
    def update(self, x: float, t: int):
        cur_region  = self._find_region(x)
        cur_step = self.t0 / max(self.t0, t)
        self.theta -= cur_step * self.region_probs
        self.theta[cur_region] += cur_step
        
        # Control size of theta to avoid overflow. Doesn't affect algorithm otherwise
        self.theta -= np.max(self.theta)
        self.theta_visits[cur_region] += 1
    
    def calc_estimator(self, x_array):
        empty_regions = (
            self.theta_visits == 0
        )
        m0 = np.sum(empty_regions)
        pi0 = np.sum(
            self.region_probs[empty_regions] / (self.num_regions - m0)
        )
        # pi0 = 0
        ref = np.max(self.theta)
        mass = np.exp(self.theta - ref) * (self.region_probs + pi0)
        return mass[-1] / np.sum(mass)