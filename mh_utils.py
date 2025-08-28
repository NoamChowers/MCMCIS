from abc import ABC, abstractmethod
from typing import Callable, List, Union, Tuple
import numpy as np
import math

from variance_estimation import MCMCVarianceEstimation


class PermutationStatistic(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def __call__(self, s1, s2) -> float:
        raise NotImplementedError("Must be implemented!")


class PermutationPropsal(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def __call__(self, s1, s2) -> Tuple[np.array, np.array]:
        raise NotImplementedError("Must be implemented!")



class AdaptiveImportanceSampling(ABC):
    def __init__(
        self,
        importance_pdf: Callable,
        log_importance_pdf: Callable,
        target_pdf: Callable,
        update_after_each_sample:bool
    ):
        self.log_importance_pdf = log_importance_pdf
        self.importance_pdf = importance_pdf
        self.target_pdf = target_pdf
        self.update_after_each_sample = update_after_each_sample
        self.params_initialized = False
    
    def calc_weight(self, x):
        return self.target_pdf(x) / self.importance_pdf(x)
    
    def calc_log_weight(self, x):
        return np.log(self.target_pdf(x)) - self.calc_log_importance_pdf(x)
    
    def calc_importance_pdf(self, x) -> float:
        """
        Evaluate unnormalized PDF
        """
        return self.importance_pdf(x)
    
    def calc_log_importance_pdf(self, x) -> float:
        """
        Evaluate unnormalized PDF
        """
        return self.log_importance_pdf(x)

    
    @abstractmethod
    def initialize_parameters(self, *args, **kwargs):
        """
        Called once after first run to initialize parameters
        (e.g., partitioning λ or setting β_0, η_0, etc.)
        """
        self.params_initialized = True

    @abstractmethod
    def update(self, x: Union[float, np.array], t):
        """
        Adaptive update. If update is called after every sample, x will be the current state of the chain.
        Otherwise (called after T+B iterations), x will be the array of all samples
        """
        pass
    
    @abstractmethod
    def calc_estimator(self, x_array):
        pass


class MCMCISPermutationTest(ABC):
    """
    An Abstract class to apply an Adaptive MH MCMC for a 2-sample permutation test.
    Can be customized for SAMC / Shuli's Algorithm / Any other procedure of type
    1) We assume permutations under H0 are Uniformly likely
    2) We assume the p-value is a one-sided test for the right tail of the distribution
    """
    def __init__(
        self,
        s1: np.array,
        s2: np.array,
        J: int,  # Number of adaptation iterations
        T: int,  # Number of steps per adaptation iteration
        B: int,  # Burn-in
        statistic_func: PermutationStatistic,  # Receives arguments s1, s2 and returns a scalar
        proposal_func: PermutationPropsal, # Receives a permutation and returns a proposal (s1, s2)
        ais: AdaptiveImportanceSampling,  # An instance of Importance function
        variance_estimation_method: MCMCVarianceEstimation,
        batch_size: Callable = lambda n: np.floor(np.sqrt(n)),  # Given n, decide on that batch size
        log_scale: bool = True,  # This helps avoid stack overflow
        seed: int = 42,
        verbose: bool = True,
        calc_estimator_variance: bool = False,
        initial_perm: Union[Tuple[np.array, np.array], None] = None,
    ):
        self.s = np.concatenate((s1, s2))
        self.n1 = len(s1)
        self.statistic_func = statistic_func
        self.lambda_star = self.statistic_func(s1, s2)
        self.proposal_func = proposal_func
        self.variance_estimation_method = variance_estimation_method
        self.batch_size = batch_size
        self.calc_estimator_variance = calc_estimator_variance
        if self.calc_estimator_variance:
            self.var_array = []
        
        # Initialize Importance Sampling instance
        self.ais = ais 
        if not self.ais.params_initialized:
            self.ais.initialize_parameters()

        if log_scale:
            self.accept_proposal = lambda cur, prop: np.log(np.random.uniform(0, 1)) < (prop - cur)
            self.importance_pdf = lambda x: self.ais.calc_log_importance_pdf(x)
            # self.weight = self.ais.calc_log_weight
        else:
            self.accept_proposal = lambda cur, prop: np.random.uniform(0, 1) < prop / cur
            self.importance_pdf = lambda x: self.ais.calc_importance_pdf(x)
        
        self.weight = self.ais.calc_weight

        self.verbose = verbose
        self.J = J
        self.T = T
        self.B = B
        self.estimator_array = []
        self.seed = seed
        self.initial_perm = initial_perm
    
    
    def update_chain_state(self, burn_in: bool):
        # Calculate proposal
        prop_s1, prop_s2 = self.proposal_func(self.cur_s1, self.cur_s2)
        prop_lambda = self.statistic_func(prop_s1, prop_s2)
        prop_importance_val = self.importance_pdf(prop_lambda)
        # Accept/Reject proposal
        if self.accept_proposal(self.cur_importance_val, prop_importance_val):
            self.cur_importance_val = prop_importance_val
            self.cur_s1, self.cur_s2, self.cur_lambda = prop_s1, prop_s2, prop_lambda
            if not burn_in:
                self.accepted += 1
        if not burn_in:
            self.lambda_array.append(self.cur_lambda)
            self.weights_array.append(
                self.weight(self.cur_importance_val)
            )


    def run_chain(self):
        np.random.seed(self.seed)
        
        if self.initial_perm is not None:
            self.cur_s1, self.cur_s2 = self.initial_perm[0], self.initial_perm[1]
        else:  # Random initialization
            init_s = np.random.permutation(self.s)
            self.cur_s1 = init_s[:self.n1]
            self.cur_s2 = init_s[self.n1:]
        self.cur_lambda = self.statistic_func(self.cur_s1, self.cur_s2)
        self.cur_importance_val = self.importance_pdf(self.cur_lambda)

        for j in range(self.J):
            if self.verbose:
                print(f"### Starting Adaptation Chain {j+1}/{self.J} ###")
            
            self.lambda_array, self.weights_array = [], []
            self.accepted = 0
            
            for i in range(self.T + self.B):
                self.update_chain_state(burn_in = i < self.B)
                if self.ais.update_after_each_sample and i >= self.B:  # For algorithms such as SAMC
                    self.ais.update(self.cur_lambda, i)

            if self.calc_estimator_variance:
                variance = self.variance_estimation_method(
                    U = self.weights_array * (self.lambda_array >= self.lambda_star).astype(int),
                    V = self.weights_array
                )
                self.var_array.append(variance)

            self.estimator_array.append(
                self.ais.calc_estimator(self.lambda_array)
            )
            
            if self.verbose:
                print(f"p-value = {self.estimator_array[-1]}")
                if self.calc_estimator_variance:
                    print(f"SD estimate = {np.sqrt(self.var_array[-1])}")
                print(f"Acceptance rate = {round(self.accepted / self.T, 4)}")
            
            if not self.ais.update_after_each_sample:
                self.ais.update(self.lambda_array[-self.T:], i)
        
        return self
