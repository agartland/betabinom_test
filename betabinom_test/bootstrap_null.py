import numpy as np

def _bootstrap_from_null(samp_inds, model_class, formula, data, null_formula, exog_test_params, count_col):
    #model_class = eval(model_name)

    np.random.seed(samp_inds[0])

    model = model_class()
    res = model.fit(formula=formula, data=data, null_formula=null_formula)

    res.fit_null(null_formula)

    coef_ind = [model.exog.design_info.column_names.index(ep) for ep in exog_test_params]

    """Simulate data under the null hypothesis [nsamps, data rows]"""
    nsamps = len(samp_inds)
    Wsamples = model.null_res.random_samples(size=nsamps)
    param_samples = np.zeros((nsamps, len(exog_test_params)))
    
    for i, si in enumerate(samp_inds):
        """Fit each null dataset with the actual covars"""
        model.data.loc[:, count_col] = Wsamples[i, :]
        model.fit(formula, model.data)
        param_samples[i, :] = model.mu_coefs[coef_ind]

    return param_samples