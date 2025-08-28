import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm
from mh_utils import (MCMCISPermutationTest, AdaptiveImportanceSampling, PermutationStatistic)
from typing import Callable, Union



class SmoothedMCMCIS(AdaptiveImportanceSampling):
    def __init__(
        self,
        s1: np.array,
        s2: np.array,
        lambda_star: float,
        target_prob: float,
        statistic_func: PermutationStatistic,  # Receives arguments s1, s2 and returns a scalar
        target_pdf: Callable = lambda x: 1,
        beta: float = 0.0,
        init_iterations: int = 100_000
        
    ):
        self.target_prob = target_prob
        self.lambda_star = lambda_star
        self.beta = beta
        self.init_iterations = init_iterations
        self.statistic_func = statistic_func
        
        super().__init__(self.importance_pdf, self.log_importance_pdf, target_pdf, update_after_each_sample=False)
        self.initialize_parameters(s1, s2)



    def _initial_lambda_std_estimate_by_random_sampling(self, s1, s2):
        s = np.concatenate([s1, s2])
        n1 = len(s1)
        lambda_array = []
        for _ in range(self.init_iterations):
            cur_perm = np.random.permutation(s)
            cur_s1 = cur_perm[:n1]
            cur_s2 = cur_perm[n1:]
            lambda_array.append(
                self.statistic_func(cur_s1, cur_s2)
            )

        return np.array(lambda_array).std(ddof=1)

    
    def _h_func(self, beta):
        tail = 1 - norm.cdf(self.lambda_star, 0, self.sigma_hat)
        phi1 = norm.cdf(self.lambda_star, beta * self.sigma_hat**2, self.sigma_hat)
        phi2 = norm.cdf(0, beta * self.sigma_hat, 1)
        exponential = np.exp(beta * (0.5*beta*self.sigma_hat**2 - self.lambda_star))
        body =  (phi1-phi2) * exponential
        return tail / (tail + body) - self.target_prob


    def _dh_func(self, beta):
        tail = 1 - norm.cdf(self.lambda_star, 0, self.sigma_hat)
        phi1 = norm.cdf(self.lambda_star, beta * self.sigma_hat**2, self.sigma_hat)
        phi2 = norm.cdf(0, beta * self.sigma_hat, 1)
        exponential = np.exp(beta * (0.5*beta*self.sigma_hat**2 - self.lambda_star))
        B =  (phi1-phi2) * exponential
        exponential2 = np.exp(-0.5 * self.sigma_hat**2 * beta**2) - np.exp(-0.5*((self.lambda_star - self.sigma_hat**2 * beta)**2)/self.sigma_hat**2)
        A = self.sigma_hat**2 * beta * (phi1 - phi2) + exponential2 * self.sigma_hat / np.sqrt(2* np.pi)
        dB = exponential * A - self.lambda_star * B
        return - (dB * tail) / (tail + B)**2

    
    def initialize_parameters(self, s1, s2):
        self.sigma_hat = self._initial_lambda_std_estimate_by_random_sampling(s1, s2)
        self.beta = fsolve(
            self._h_func,
            self.beta,
            # args=(self.sigma_hat, self.target_prob)
        )[0]
        super().initialize_parameters()
    
    
    def importance_pdf(self, x):    
        return np.exp(self.log_importance_pdf(x))


    def log_importance_pdf(self, x):    
        return self.beta * (x - self.lambda_star) * (x <= self.lambda_star)

    
    def update(self, x_array: np.array, *args, **kwargs):
        pi_hat = np.mean(x_array >= self.lambda_star)
        dh = self._dh_func(self.beta)
        self.beta = max(
            0,
            self.beta - (pi_hat - self.target_prob) / dh
        )
    
    def calc_estimator(self, x_array):
        weights = super().calc_weight(x_array)
        return np.sum(
            weights * (x_array >= self.lambda_star).astype(int)
        ) / np.sum(weights)
