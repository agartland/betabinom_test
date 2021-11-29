import numpy as np
import pandas as pd
from scipy import stats, special
from scipy.optimize import minimize

from os.path import join as opj
import os

from patsy import dmatrices
import parmap

from .bootstrap_null import _bootstrap_from_null

"""Create simple beta-binomial model for one binary predictor that would 
allow for fast permutation or bootstrap testing across cores

TODO:

Test that this code holds Type I error

Compare simple test data scenario (one covar) corncob vs. python with bootstrap

"""

__all__ = ['beta_binom_full',
            'beta_binom_fixed_phi']

"""Parameterization"""
# a = mu * (1/phi -1)
# b = ((mu - 1) * (phi - 1)) / mu

class base_model:
    def _logistic(x):
        """Inverse logit"""
        p = np.exp(x) / (np.exp(x) + 1)
        return p
    def _logit(p):
        x = np.log(p / (1 - p))
        return x

class beta_binom_full(base_model):
    """Counts regression model using beta-binomial distribution
        
    Martin, B. D., Witten, D., & Willis, A. D. (2020).
          MODELING MICROBIAL ABUNDANCES AND DYSBIOSIS WITH
          BETA-BINOMIAL REGRESSION. The annals of applied
          statistics, 14(1), 94–115. https://doi.org/10.1214/19-aoas1283

    Model
    -----     
    Wi | (Zi, Mi) ∼ Binomial(Mi, Zi)

    Zi ∼ Beta(a1, a2)
     
    where,
    Wi is the number of events (k),
    Mi is the total number of draws (n),
    Zi is a latent var representing the randomly drawn probability
    a1, a2 are parameters for the Beta distribution of Zi

    Model parameters
    ----------------
    mu_coefs : mean probability for the binomial
                (sum of regression coefficients X data)
    phi0 : overdispersion relative to a binomial [0, 1]
            (where phi = 0 is equivalent to a Binomial)

    (re-parameterization of a1, a2)"""

    def __init__(self):
        """coefs: real-valued parameters for likelihood function"""
        """coefs0 = initial guesses for optimization"""
        self.fit_method = 'BFGS'
        self.fit_kwargs = {'disp': False, 'gtol':1e-6}
        #self.fit_method = 'nelder-mead'
        #self.fit_kwargs = {'xatol': 1e-8, 'disp': False}
        self.self_class = beta_binom_full

    def params_to_a1a2(mu, phi):
        a1 = mu * (1/phi - 1)
        a2 = (mu - 1) * (phi - 1) / phi
        return np.array([a1, a2])
    def a1a2_to_params(a1, a2):
        mu = a1 / (a1 + a2)
        phi = 1 / (a1 + a2 + 1)

    def get_beta_params(self, mu_coefs=None, phi_coefs=None, exog=None, as_a1a2=True):
        """Fetch the parameters expressed as those for the Beta(a1, a2)

        By default computes parameters from the coefficient vectors in the object,
        which requires that the `fit` method has been called.

        With as_a1a2 = False can return the mu/phi parameterization instead"""
        if mu_coefs is None:
            mu_coefs = self.mu_coefs
        if phi_coefs is None:
            phi_coefs = self.phi_coefs
        if exog is None:
            exog = self.exog

        mu = beta_binom_full._logistic(np.dot(mu_coefs, exog.T))
        phi = beta_binom_full._logistic(phi_coefs) * np.ones(mu.shape[0])

        if as_a1a2:
            a1 = mu * (1/phi - 1)
            a2 = (mu - 1) * (phi - 1) / phi
            return np.concatenate((a1[:, None], a2[:, None]), axis=1)
        else:
            return np.concatenate((mu[:, None], phi[:, None]), axis=1)

    def predict(self, mu_coefs=None, phi_coefs=None, exog=None, M=None):
        if M is None:
            M = self.M

        mu_phi = self.get_beta_params(mu_coefs=mu_coefs, phi_coefs=phi_coefs, exog=exog, as_a1a2=False)

        What = np.squeeze(mu_phi[:, 0]) * M
        self.residuals = self.W - What
        return What

    def fit(self, formula, data, null_formula=None):
        """Fit model using formula and data provided. Can use
        a null_formula to make better guesses at the initial params.

        Formula must use 'W' and 'M' as names for the count numerator and
        denominator, respectively. The formula is parsed by `patsy.dmatrices`
        
        Parameters
        ----------
        formula : str
            Example "W + M ~ timepoint + group"
            Must use W and M as variables for numerator and denominator, respectively.
        data : pd.DataFrame [data observations x variables]
        null_formula : str
            Example "W + M ~ 1"
            Will fit the null model to get initial coef estimates for fitting (e.g. intercept)
            This may improve fitting if there are many parameters.

        Returns
        -------
        res : beta_binom_full
            Returns self, with results also assigned to the model.
        """
        self.data = data
        self.formula = formula

        """Use formula like W + M ~ covar """
        self.endog, self.exog = dmatrices(formula, data=data, NA_action='drop')

        self.W = self.endog[:, self.endog.design_info.column_names.index('W')]
        self.M = self.endog[:, self.endog.design_info.column_names.index('M')]

        self.n_mu_coefs = self.exog.shape[1]
        self.phi_coefs0 = np.zeros(1)
        self.mu_coefs0 = np.zeros(self.n_mu_coefs)

        if not null_formula is None:
            """Use the null model to improve initial guesses"""
            self.fit_null(null_formula)

            for i, col in enumerate(self.exog.design_info.column_names):
                if col in self.null_res.exog.design_info.column_names:
                    i_null = self.null_res.exog.design_info.column_names.index(col)
                    self.mu_coefs0[i] = self.null_res.mu_coefs0[i_null]        

        coefs0 = np.concatenate((self.mu_coefs0, self.phi_coefs0))
        res = minimize(beta_binom_full._linked_neg_loglik,
                         coefs0,
                         args=(self.exog, self.endog),
                         method=self.fit_method,
                         options=self.fit_kwargs)
        self.res = res
        self.mu_coefs = res.x[:self.n_mu_coefs]
        self.phi_coefs = res.x[self.n_mu_coefs:]
        return self

    def OLD_linked_neg_loglik(x, covar, k, n):
        u0, u1, phi0 = x
        a = np.exp(-phi0) / (1 + np.exp(-u0 - u1*covar))
        b = np.exp(-phi0) / (1 + np.exp(u0 + u1*covar))
        ll =  special.betaln(k + a, n - k + b) - special.betaln(a, b)
        return -np.sum(ll)

    def _linked_neg_loglik(x, exog, endog):
        """Assumes:
            Only one phi coef
            exog is design matrix
            endog contains k, n (W, M) in first two columns"""
        tmp_mu = np.dot(x[:-1], exog.T)
        a = np.exp(-x[-1]) / (1 + np.exp(-tmp_mu))
        b = np.exp(-x[-1]) / (1 + np.exp(tmp_mu))
        ll =  special.betaln(endog[:, 0] + a, endog[:, 1] - endog[:, 0] + b) - special.betaln(a, b)
        return -np.sum(ll)

    def random_samples(self, size):
        """Generate event counts based on the fitted data."""
        a1a2 = self.get_beta_params(as_a1a2=True)
        out = stats.betabinom.rvs(self.M.astype(np.int64),
                                  a1a2[:, 0],
                                  a1a2[:, 1],
                                  size=(size, a1a2.shape[0]))
        return out

    def fit_null(self, null_formula):
        """Fit a null model, useful for bootstrap testing."""
        self.null_formula = null_formula

        """Calling constructor for the class itself"""
        self.null_model = self.self_class()
        self.null_res = self.null_model.fit(null_formula, self.data) 
        return self

    def bootstrap_pvalue(self, null_formula, exog_test_params, nsamps=1000, ncpus=1):
        """Compute a pvalue for each parameter in exog_test_params.
        Process is to fit a null model and draw random datasets from it.
        Then fit each random dataset as a full model and report proportion of coefs as extreme
        as the actual coefs.

        Parameters
        ----------
        null_formula : str
            Of form "W + M ~ 1" or include other covars that are part of the null model
        exog_test_params : list
            List of variable names in fit formula/data (probably not in null_formula)
            that will be tested against the null hypothesis.
        nsamps : int
            Number of samples for simulation.
        ncpus : int
            Number of CPUs for multiprocessing parallelization

        Returns
        -------
        pvalues : np.ndarray
            Vector of p-values with length equal to number of exog_test_variables
        """

        """Currently this is hardcoded because it is a requirement of the formula"""
        count_col = 'W'

        self.nsamps = nsamps

        coef_ind = [self.exog.design_info.column_names.index(ep) for ep in exog_test_params]

        kwargs = dict(model_class=self.self_class,
                      formula=self.formula,
                      data=self.data,
                      null_formula=null_formula,
                      exog_test_params=exog_test_params,
                      count_col=count_col)
        if ncpus == 1:
            param_samples = _bootstrap_from_null(range(nsamps), **kwargs)
        else:
            chunk_func = lambda l, n: [l[i:i + n] for i in range(0, len(l), n)]
            chunksz = nsamps//ncpus
            tmp = parmap.map(_bootstrap_from_null,
                                 chunk_func(list(range(nsamps)), chunksz),
                                 pm_processes=ncpus,
                                 **kwargs)

            param_samples = np.concatenate(tmp)

        """What proportion of null datasets have a more extreme mu coef?"""
        """This may return multiple pvalues if multiple test columns are specified"""
        pvalues = (np.sum(np.abs(param_samples) >= np.abs(self.mu_coefs[coef_ind]), axis=0) + 1) / (nsamps + 1)
        self.pvalues = pvalues
        self.param_samples = param_samples
        #self.exog_test_params = exog_test_params
        #self.exog_test_ind = coef_ind
        return pvalues
