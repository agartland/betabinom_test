"""
python -m pytest betabinom_test/tests/test_model.py
"""
import unittest
import numpy as np
import pandas as pd

from scipy import stats

from betabinom_test import beta_binom_full, beta_binom_fixed_phi

def _simple_data():
    np.random.seed(110820)
    L = 100
    M = 500 * np.ones(L, dtype=np.int64)
    a1, a2 = beta_binom_full.params_to_a1a2(mu=0.5, phi=0.4)
    W = stats.betabinom.rvs(M, a1, a2, size=L)
    covar = np.random.randint(2, size=L)

    """Note that phi must be passed as coef (ie logit(phi))"""
    df = pd.DataFrame({'covar':covar,
                            'W':W,
                            'M':M,
                            'phi':np.ones(L) * beta_binom_full._logit(0.005)})
    return df

class TestBB(unittest.TestCase):

    def test_fit(self):
        df = _simple_data()
        mod = beta_binom_full()
        res = mod.fit(formula='W + M ~ covar',
                        data=df)

    def test_fit_with_null(self):
        df = _simple_data()
        mod = beta_binom_full()
        res = mod.fit(formula='W + M ~ covar',
                        data=df,
                        null_formula='W + M ~ 1')

        pvalues = res.bootstrap_pvalue(null_formula='W + M ~ 1',
                                       exog_test_params=['covar'],
                                       nsamps=100)
    def test_bootstrap(self, nsamps=100):
        df = _simple_data()
        mod = beta_binom_full()
        res = mod.fit(formula='W + M ~ covar',
                        data=df,
                        null_formula='W + M ~ 1')

        pvalues = res.bootstrap_pvalue(null_formula='W + M ~ 1',
                                       exog_test_params=['covar'],
                                       nsamps=nsamps)

    def test_bootstrap_parallel(self, nsamps=100):
        df = _simple_data()
        mod = beta_binom_full()
        res = mod.fit(formula='W + M ~ covar',
                        data=df,
                        null_formula='W + M ~ 1')

        pvalues = res.bootstrap_pvalue(null_formula='W + M ~ 1',
                                       exog_test_params=['covar'],
                                       nsamps=nsamps,
                                       ncpus=2)
    def test_predict(self):
        df = _simple_data()
        mod = beta_binom_full()
        res = mod.fit(formula='W + M ~ covar',
                        data=df,
                        null_formula='W + M ~ 1')
        What = res.predict()

        df = df.assign(What=What,
                       resid=df['W']- What)

class TestBBFixedPhi(unittest.TestCase):

    def test_fit(self):
        df = _simple_data()
        mod = beta_binom_fixed_phi()
        res = mod.fit(formula='W + M + phi ~ covar',
                        data=df)

if __name__ == '__main__':
    unittest.main()
