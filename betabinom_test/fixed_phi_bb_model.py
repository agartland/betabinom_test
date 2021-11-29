import numpy as np
import pandas as pd
from scipy import stats, special
from scipy.optimize import minimize

from os.path import join as opj
import os

from patsy import dmatrices
import parmap

from .bootstrap_null import _bootstrap_from_null
from .model import beta_binom_full

__all__ = ['beta_binom_fixed_phi']

class beta_binom_fixed_phi(beta_binom_full):
    """Counts regression model using beta-binomial distribution with a
    fixed phi parameter supplied in the data
        
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
        self.self_class = beta_binom_fixed_phi
        super().__init__()

    def fit(self, formula, data, null_formula=None):
        """Fit model using formula and data provided. Can use
        a null_formula to make better guesses at the initial params.

        Formula must use 'W' and 'M' as names for the count numerator and
        denominator, respectively. The formula is parsed by `patsy.dmatrices`

        'phi' should also be pass as data and will be used as a coef (post logit)
        This makes it possible to provide a phi for each data observation,
        but in practice it should probably be the same for all observations.

        A phi = 0 is equivalent to a binomial (phi_coef < -10)
        
        Parameters
        ----------
        formula : str
            Example "W + M + phi ~ timepoint + group"
            Must use W, M, phi as variables for numerator, denominator, and phi_coef respectively.
        data : pd.DataFrame [data observations x variables]
        null_formula : str
            Example "W + M + phi ~ 1"
            Will fit the null model to get initial coef estimates for fitting (e.g. intercept)
            This may improve fitting if there are many parameters.

        Returns
        -------
        res : beta_binom_full
            Returns self, with results also assigned to the model.
        """
        self.data = data
        self.formula = formula

        self.endog, self.exog = dmatrices(formula, data=data, NA_action='drop')

        self.W = self.endog[:, self.endog.design_info.column_names.index('W')]
        self.M = self.endog[:, self.endog.design_info.column_names.index('M')]
        self.phi_coefs = self.endog[:, self.endog.design_info.column_names.index('phi')]

        self.n_mu_coefs = self.exog.shape[1]
        self.mu_coefs0 = np.zeros(self.n_mu_coefs)

        if not null_formula is None:
            """Use the null model to improve initial guesses"""
            self.fit_null(null_formula)

            for i, col in enumerate(self.exog.design_info.column_names):
                if col in self.null_res.exog.design_info.column_names:
                    i_null = self.null_res.exog.design_info.column_names.index(col)
                    self.mu_coefs0[i] = self.null_res.mu_coefs0[i_null]        

        res = minimize(beta_binom_fixed_phi._linked_neg_loglik,
                         self.mu_coefs0,
                         args=(self.exog, self.endog),
                         method=self.fit_method,
                         options=self.fit_kwargs)
        self.res = res
        self.mu_coefs = res.x
        return self

    def _linked_neg_loglik(x, exog, endog):
        """Assumes:
            exog is design matrix
            endog contains k, n (W, M) in first two columns
            endog contains phi in thrid column"""
        tmp_mu = np.dot(x, exog.T)
        a = np.exp(-endog[:, 2]) / (1 + np.exp(-tmp_mu))
        b = np.exp(-endog[:, 2]) / (1 + np.exp(tmp_mu))
        ll =  special.betaln(endog[:, 0] + a, endog[:, 1] - endog[:, 0] + b) - special.betaln(a, b)
        return -np.sum(ll)