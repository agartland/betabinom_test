# betabinom_test

[![Build Status](https://travis-ci.com/agartland/betabinom_test.svg?branch=main)](https://travis-ci.com/agartland/betabinom_test)
[![PyPI version](https://badge.fury.io/py/betabinom_test.svg)](https://badge.fury.io/py/betabinom_test)
[![Coverage Status](https://coveralls.io/repos/github/agartland/betabinom_test/badge.svg?branch=main)](https://coveralls.io/github/agartland/betabinom_test?branch=main)

Package for fitting and performing basic hypothesis testing with a beta-binomial count regression model. P-values are generated from a bootstrap method that simulates coefficients from a null model.

## Installation

```
pip install betabinom_test
```

## Example

```python
from betabinom_test import beta_binom_full

L = 100
M = 500 * np.ones(L, dtype=np.int64)
a1, a2 = beta_binom_full.params_to_a1a2(mu=0.5, phi=0.4)
W = stats.betabinom.rvs(M, a1, a2, size=L)
covar = np.random.randint(2, size=L)

df = pd.DataFrame({'covar':covar,
                        'W':W,
                        'M':M})

mod = beta_binom_full()

res = mod.fit(formula='W + M ~ covar',
              data=df,
              null_formula='W + M ~ 1')

pvalues = res.bootstrap_pvalue(null_formula='W + M ~ 1',
                               exog_test_params=['covar'],
                               nsamps=1000,
                               ncpus=2)
```