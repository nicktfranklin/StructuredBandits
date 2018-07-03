import numpy as np
import pandas as pd

import pymc3 as pm
from theano.tensor.nnet.nnet import softmax
from theano import tensor as tt
import theano as t

print('Runing on PyMC3 v{}'.format(pm.__version__))
print('Runing on Theano v{}'.format(t.__version__))


def sample_hier_rbf(model_matrix, sample_kwargs=None):

    # load the data
    x_mu_rbf = model_matrix['x_mu_rbf']
    x_sd_rbf = model_matrix['x_sd_rbf']
    x_sc = model_matrix['x_sc']
    subj_idx = model_matrix['subj_idx']
    y = model_matrix['y']
    n_subj = model_matrix['n_subj']

    # fit the first model
    n, d = x_mu_rbf.shape
    if sample_kwargs is None:
        # Here, we use specify NUTS as our sampler (implicitly this is the default)
        # and use variational inference to initialize
        sample_kwargs = dict(draws=2000, njobs=2, tune=2000, init='advi+adapt_diag')

    # to do inference, all we have to do is write down the model in our
    # probabilistic programming language (PYMC3) and the software will
    # do inference over it (we can control how this happens, e.g. with
    # Gibbs sampling, MCMC, Variational Inference, but PYMC3 will default
    # to hamiltonian-MCMC with the No U-turn sampler ("NUTS"))

    with pm.Model() as hier_rbf:
        # here, we write down the model

        # Define hierarchical parameters
        # (normal means and standard deviation for regression weights)
        mu_1 = pm.Normal('mu_beta_rbf_mean', mu=0., sd=100.)
        mu_2 = pm.Normal('mu_beta_rbf_stdv', mu=0., sd=100.)
        mu_3 = pm.Normal('mu_beta_stick',    mu=0., sd=100.)

        sigma_1 = pm.HalfCauchy('sigma_rbf_means', beta=100)
        sigma_2 = pm.HalfCauchy('sigma_rbf_stdev', beta=100)
        sigma_3 = pm.HalfCauchy('sigma_stick',     beta=100)

        # define subject predictor variables (i.e. regression parameters,
        # 1 per subject per condition with a hierarchical prior)
        b_1 = pm.Normal('beta_rbf_mu',  mu=mu_1, sd=sigma_1, shape=n_subj)
        b_2 = pm.Normal('beta_rbf_std', mu=mu_2, sd=sigma_2, shape=n_subj)
        b_3 = pm.Normal('beta_sc',      mu=mu_3, sd=sigma_3, shape=n_subj)

        # linearly combine the predictors with the subject-specific coefficients
        # as a scaling factor. In practice, the coefficients have to be broadcast
        # in to an NxD matric via theano for element-wise multiplication
        rho = \
            tt.tile(tt.reshape(b_1[subj_idx], (n, 1)), d) * x_mu_rbf + \
            tt.tile(tt.reshape(b_2[subj_idx], (n, 1)), d) * x_sd_rbf + \
            tt.tile(tt.reshape(b_3[subj_idx], (n, 1)), d) * x_sc

        # pass the resultant vector through a softmax to convert to a probability
        # distribution. Note, we don't need an additional noise parameter as that
        # would be collinear with the coefficients.
        p_hat = softmax(rho)

        # Data likelihood
        yl = pm.Categorical('yl', p=p_hat, observed=y)

        # inference!
        trace_rbf = pm.sample(**sample_kwargs)

    return hier_rbf, trace_rbf


def sample_hier_lin(model_matrix, sample_kwargs=None):

    # load the data
    x_mu_lin = model_matrix['x_mu_lin']
    x_sd_lin = model_matrix['x_sd_lin']
    x_sc = model_matrix['x_sc']
    subj_idx = model_matrix['subj_idx']
    y = model_matrix['y']
    n_subj = model_matrix['n_subj']

    n, d = x_mu_lin.shape
    if sample_kwargs is None:
        sample_kwargs = dict(draws=2000, njobs=2, tune=2000, init='advi+adapt_diag')

    with pm.Model() as hier_lin:
        mu_1 = pm.Normal('mu_beta_lin_mean', mu=0., sd=100.)
        mu_2 = pm.Normal('mu_beta_lin_stdv', mu=0., sd=100.)
        mu_3 = pm.Normal('mu_beta_stick',    mu=0., sd=100.)

        sigma_1 = pm.HalfCauchy('sigma_lin_means', beta=100)
        sigma_2 = pm.HalfCauchy('sigma_lin_stdev', beta=100)
        sigma_3 = pm.HalfCauchy('sigma_stick',     beta=100)

        b_1 = pm.Normal('beta_lin_mu',  mu=mu_1, sd=sigma_1, shape=n_subj)
        b_2 = pm.Normal('beta_lin_std', mu=mu_2, sd=sigma_2, shape=n_subj)
        b_3 = pm.Normal('beta_sc',      mu=mu_3, sd=sigma_3, shape=n_subj)

        rho = \
            tt.tile(tt.reshape(b_1[subj_idx], (n, 1)), d) * x_mu_lin + \
            tt.tile(tt.reshape(b_2[subj_idx], (n, 1)), d) * x_sd_lin + \
            tt.tile(tt.reshape(b_3[subj_idx], (n, 1)), d) * x_sc

        p_hat = softmax(rho)

        # Data likelihood
        yl = pm.Categorical('yl', p=p_hat, observed=y)

        # inference!
        trace_lin = pm.sample(**sample_kwargs)

    return hier_lin, trace_lin


def sample_hier_cls(model_matrix, sample_kwargs=None):

    # load the data
    x_mu_cls = model_matrix['x_mu_cls']
    x_sd_cls = model_matrix['x_sd_cls']
    x_sc = model_matrix['x_sc']
    subj_idx = model_matrix['subj_idx']
    y = model_matrix['y']
    n_subj = model_matrix['n_subj']

    n, d = x_mu_cls.shape
    if sample_kwargs is None:
        sample_kwargs = dict(draws=2000, njobs=2, tune=2000, init='advi+adapt_diag')

    with pm.Model() as hier_cls:
        mu_1 = pm.Normal('mu_beta_cls_mean', mu=0., sd=100.)
        mu_2 = pm.Normal('mu_beta_cls_stdv', mu=0., sd=100.)
        mu_3 = pm.Normal('mu_beta_stick',    mu=0., sd=100.)

        sigma_1 = pm.HalfCauchy('sigma_cls_means', beta=100)
        sigma_2 = pm.HalfCauchy('sigma_cls_stdev', beta=100)
        sigma_3 = pm.HalfCauchy('sigma_stick',     beta=100)

        b_1 = pm.Normal('beta_cls_mu',  mu=mu_1, sd=sigma_1, shape=n_subj)
        b_2 = pm.Normal('beta_cls_std', mu=mu_2, sd=sigma_2, shape=n_subj)
        b_3 = pm.Normal('beta_sc',      mu=mu_3, sd=sigma_3, shape=n_subj)

        rho = \
            tt.tile(tt.reshape(b_1[subj_idx], (n, 1)), d) * x_mu_cls + \
            tt.tile(tt.reshape(b_2[subj_idx], (n, 1)), d) * x_sd_cls + \
            tt.tile(tt.reshape(b_3[subj_idx], (n, 1)), d) * x_sc

        p_hat = softmax(rho)

        # Data likelihood
        yl = pm.Categorical('yl', p=p_hat, observed=y)

        # inference!
        trace_cls = pm.sample(**sample_kwargs)

    return hier_cls, trace_cls

def sample_hier_kal(model_matrix, sample_kwargs=None):

    # load the data
    x_mu_kal = model_matrix['x_mu_kal']
    x_sd_kal = model_matrix['x_sd_kal']
    x_sc = model_matrix['x_sc']
    subj_idx = model_matrix['subj_idx']
    y = model_matrix['y']
    n_subj = model_matrix['n_subj']

    n, d = x_mu_kal.shape
    if sample_kwargs is None:
        sample_kwargs = dict(draws=2000, njobs=2, tune=2000, init='advi+adapt_diag')

    with pm.Model() as hier_kal:
        mu_1 = pm.Normal('mu_beta_kal_mean', mu=0., sd=100.)
        mu_2 = pm.Normal('mu_beta_kal_stdv', mu=0., sd=100.)
        mu_3 = pm.Normal('mu_beta_stick',    mu=0., sd=100.)

        sigma_1 = pm.HalfCauchy('sigma_rbf_means', beta=100)
        sigma_2 = pm.HalfCauchy('sigma_rbf_stdev', beta=100)
        sigma_3 = pm.HalfCauchy('sigma_stick',     beta=100)

        b_1 = pm.Normal('beta_rbf_mu',  mu=mu_1, sd=sigma_1, shape=n_subj)
        b_2 = pm.Normal('beta_rbf_std', mu=mu_2, sd=sigma_2, shape=n_subj)
        b_3 = pm.Normal('beta_sc',      mu=mu_3, sd=sigma_3, shape=n_subj)

        rho = \
            tt.tile(tt.reshape(b_1[subj_idx], (n, 1)), d) * x_mu_kal + \
            tt.tile(tt.reshape(b_2[subj_idx], (n, 1)), d) * x_sd_kal + \
            tt.tile(tt.reshape(b_3[subj_idx], (n, 1)), d) * x_sc
        p_hat = softmax(rho)

        # Data likelihood
        yl = pm.Categorical('yl', p=p_hat, observed=y)

        # inference!
        trace_kal = pm.sample(**sample_kwargs)

    return hier_kal, trace_kal

def sample_hier_bayes_gp(model_matrix, sample_kwargs=None):

    # load the data
    x_mu_bayes_gp = model_matrix['x_mu_bayes_gp']
    x_sd_bayes_gp = model_matrix['x_sd_bayes_gp']
    x_sc = model_matrix['x_sc']
    subj_idx = model_matrix['subj_idx']
    y = model_matrix['y']
    n_subj = model_matrix['n_subj']

    n, d = x_mu_bayes_gp.shape
    if sample_kwargs is None:
        sample_kwargs = dict(draws=2000, njobs=2, tune=2000, init='advi+adapt_diag')

    with pm.Model() as heir_bayes_gp:
        mu_1 = pm.Normal('mu_beta_bgp_mean', mu=0., sd=100.)
        mu_2 = pm.Normal('mu_beta_bgp_stdv', mu=0., sd=100.)
        mu_3 = pm.Normal('mu_beta_stick',    mu=0., sd=100.)

        sigma_1 = pm.HalfCauchy('sigma_bgp_means', beta=100)
        sigma_2 = pm.HalfCauchy('sigma_bgp_stdev', beta=100)
        sigma_3 = pm.HalfCauchy('sigma_stick',     beta=100)

        b_1 = pm.Normal('beta_bgp_mu',  mu=mu_1, sd=sigma_1, shape=n_subj)
        b_2 = pm.Normal('beta_bgp_std', mu=mu_2, sd=sigma_2, shape=n_subj)
        b_3 = pm.Normal('beta_sc',      mu=mu_3, sd=sigma_3, shape=n_subj)

        rho = \
            tt.tile(tt.reshape(b_1[subj_idx], (n, 1)), d) * x_mu_bayes_gp + \
            tt.tile(tt.reshape(b_2[subj_idx], (n, 1)), d) * x_sd_bayes_gp + \
            tt.tile(tt.reshape(b_3[subj_idx], (n, 1)), d) * x_sc

        p_hat = softmax(rho)

        # Data likelihood
        yl = pm.Categorical('yl', p=p_hat, observed=y)

        # inference!
        trace_bayes_gp = pm.sample(**sample_kwargs)

    return heir_bayes_gp, trace_bayes_gp

def sample_hier_rbf_cls(model_matrix, sample_kwargs=None):

    # load the data
    x_mu_rbf = model_matrix['x_mu_rbf']
    x_sd_rbf = model_matrix['x_sd_rbf']
    x_mu_cls = model_matrix['x_mu_cls']
    x_sd_cls = model_matrix['x_sd_cls']
    x_sc = model_matrix['x_sc']
    subj_idx = model_matrix['subj_idx']
    y = model_matrix['y']
    n_subj = model_matrix['n_subj']

    n, d = x_mu_cls.shape
    if sample_kwargs is None:
        sample_kwargs = dict(draws=2000, njobs=2, tune=2000, init='advi+adapt_diag')

    with pm.Model() as hier_rbf_clus:
        mu_1 = pm.Normal('mu_beta_rbf_mean', mu=0., sd=100.)
        mu_2 = pm.Normal('mu_beta_rbf_stdv', mu=0., sd=100.)
        mu_3 = pm.Normal('mu_beta_cls_mean', mu=0., sd=100.)
        mu_4 = pm.Normal('mu_beta_cls_stdv', mu=0., sd=100.)
        mu_5 = pm.Normal('mu_beta_stick',    mu=0., sd=100.)

        sigma_1 = pm.HalfCauchy('sigma_rbf_means', beta=100)
        sigma_2 = pm.HalfCauchy('sigma_rbf_stdev', beta=100)
        sigma_3 = pm.HalfCauchy('sigma_cls_means', beta=100)
        sigma_4 = pm.HalfCauchy('sigma_cls_stdev', beta=100)
        sigma_5 = pm.HalfCauchy('sigma_stick',     beta=100)

        b_1 = pm.Normal('beta_rbf_mu',  mu=mu_1, sd=sigma_1, shape=n_subj)
        b_2 = pm.Normal('beta_rbf_std', mu=mu_2, sd=sigma_2, shape=n_subj)
        b_3 = pm.Normal('beta_cls_mu',  mu=mu_3, sd=sigma_3, shape=n_subj)
        b_4 = pm.Normal('beta_cls_std', mu=mu_4, sd=sigma_4, shape=n_subj)
        b_5 = pm.Normal('beta_sc',      mu=mu_5, sd=sigma_5, shape=n_subj)

        rho = \
            tt.tile(tt.reshape(b_1[subj_idx], (n, 1)), d) * x_mu_rbf + \
            tt.tile(tt.reshape(b_2[subj_idx], (n, 1)), d) * x_sd_rbf + \
            tt.tile(tt.reshape(b_3[subj_idx], (n, 1)), d) * x_mu_cls + \
            tt.tile(tt.reshape(b_4[subj_idx], (n, 1)), d) * x_sd_cls + \
            tt.tile(tt.reshape(b_5[subj_idx], (n, 1)), d) * x_sc

        p_hat = softmax(rho)

        # Data likelihood
        yl = pm.Categorical('yl', p=p_hat, observed=y)

        # inference!
        trace_gprbf_cls = pm.sample(**sample_kwargs)

    return hier_rbf_clus, trace_gprbf_cls


def sample_heir_lin_cls(model_matrix, sample_kwargs=None):

    # load the data
    x_mu_lin = model_matrix['x_mu_lin']
    x_sd_lin = model_matrix['x_sd_lin']
    x_mu_cls = model_matrix['x_mu_cls']
    x_sd_cls = model_matrix['x_sd_cls']
    x_sc = model_matrix['x_sc']
    subj_idx = model_matrix['subj_idx']
    y = model_matrix['y']
    n_subj = model_matrix['n_subj']

    n, d = x_mu_lin.shape
    if sample_kwargs is None:
        sample_kwargs = dict(draws=2000, njobs=2, tune=2000, init='advi+adapt_diag')

    with pm.Model() as hier_lin_clus:
        mu_1 = pm.Normal('mu_beta_lin_mean', mu=0., sd=100.)
        mu_2 = pm.Normal('mu_beta_lin_stdv', mu=0., sd=100.)
        mu_3 = pm.Normal('mu_beta_cls_mean', mu=0., sd=100.)
        mu_4 = pm.Normal('mu_beta_cls_stdv', mu=0., sd=100.)
        mu_5 = pm.Normal('mu_beta_stick',    mu=0., sd=100.)

        sigma_1 = pm.HalfCauchy('sigma_lin_means', beta=100)
        sigma_2 = pm.HalfCauchy('sigma_lin_stdev', beta=100)
        sigma_3 = pm.HalfCauchy('sigma_cls_means', beta=100)
        sigma_4 = pm.HalfCauchy('sigma_cls_stdev', beta=100)
        sigma_5 = pm.HalfCauchy('sigma_stick',     beta=100)

        b_1 = pm.Normal('beta_lin_mu',  mu=mu_1, sd=sigma_1, shape=n_subj)
        b_2 = pm.Normal('beta_lin_std', mu=mu_2, sd=sigma_2, shape=n_subj)
        b_3 = pm.Normal('beta_cls_mu',  mu=mu_3, sd=sigma_3, shape=n_subj)
        b_4 = pm.Normal('beta_cls_std', mu=mu_4, sd=sigma_4, shape=n_subj)
        b_5 = pm.Normal('beta_sc',      mu=mu_5, sd=sigma_5, shape=n_subj)

        rho = \
            tt.tile(tt.reshape(b_1[subj_idx], (n, 1)), d) * x_mu_lin + \
            tt.tile(tt.reshape(b_2[subj_idx], (n, 1)), d) * x_sd_lin + \
            tt.tile(tt.reshape(b_3[subj_idx], (n, 1)), d) * x_mu_cls + \
            tt.tile(tt.reshape(b_4[subj_idx], (n, 1)), d) * x_sd_cls + \
            tt.tile(tt.reshape(b_5[subj_idx], (n, 1)), d) * x_sc

        p_hat = softmax(rho)

        # Data likelihood
        yl = pm.Categorical('yl', p=p_hat, observed=y)

        # inference!
        trace_gplin_cls = pm.sample(**sample_kwargs)

    return hier_lin_clus, trace_gplin_cls


def sample_heir_rbf_kal(model_matrix, sample_kwargs=None):

    # load the data
    x_mu_rbf = model_matrix['x_mu_rbf']
    x_sd_rbf = model_matrix['x_sd_rbf']
    x_mu_kal = model_matrix['x_mu_kal']
    x_sd_kal = model_matrix['x_sd_kal']
    x_sc = model_matrix['x_sc']
    subj_idx = model_matrix['subj_idx']
    y = model_matrix['y']
    n_subj = model_matrix['n_subj']

    n, d = x_mu_rbf.shape
    if sample_kwargs is None:
        sample_kwargs = dict(draws=2000, njobs=2, tune=2000, init='advi+adapt_diag')

    with pm.Model() as hier_rbf_kal:

        mu_1 = pm.Normal('mu_beta_rbf_mean', mu=0., sd=100.)
        mu_2 = pm.Normal('mu_beta_rbf_stdv', mu=0., sd=100.)
        mu_3 = pm.Normal('mu_beta_kal_mean', mu=0., sd=100.)
        mu_4 = pm.Normal('mu_beta_kal_stdv', mu=0., sd=100.)
        mu_5 = pm.Normal('mu_beta_stick',    mu=0., sd=100.)

        sigma_1 = pm.HalfCauchy('sigma_rbf_means', beta=100)
        sigma_2 = pm.HalfCauchy('sigma_rbf_stdev', beta=100)
        sigma_3 = pm.HalfCauchy('sigma_kal_means', beta=100)
        sigma_4 = pm.HalfCauchy('sigma_kal_stdev', beta=100)
        sigma_5 = pm.HalfCauchy('sigma_stick',     beta=100)

        b_1 = pm.Normal('beta_rbf_mu',  mu=mu_1, sd=sigma_1, shape=n_subj)
        b_2 = pm.Normal('beta_rbf_std', mu=mu_2, sd=sigma_2, shape=n_subj)
        b_3 = pm.Normal('beta_kal_mu',  mu=mu_3, sd=sigma_3, shape=n_subj)
        b_4 = pm.Normal('beta_kal_std', mu=mu_4, sd=sigma_4, shape=n_subj)
        b_5 = pm.Normal('beta_sc',      mu=mu_5, sd=sigma_5, shape=n_subj)

        rho = \
            tt.tile(tt.reshape(b_1[subj_idx], (n, 1)), d) * x_mu_rbf + \
            tt.tile(tt.reshape(b_2[subj_idx], (n, 1)), d) * x_sd_rbf + \
            tt.tile(tt.reshape(b_3[subj_idx], (n, 1)), d) * x_mu_kal + \
            tt.tile(tt.reshape(b_4[subj_idx], (n, 1)), d) * x_sd_kal + \
            tt.tile(tt.reshape(b_5[subj_idx], (n, 1)), d) * x_sc

        p_hat = softmax(rho)

        # Data likelihood
        yl = pm.Categorical('yl', p=p_hat, observed=y)

        # inference!
        trace_gprbf_kal = pm.sample(**sample_kwargs)

    return hier_rbf_kal, trace_gprbf_kal


def sample_heir_scram_kal(model_matrix, sample_kwargs=None):

    # load the data + scramble Kalman filter data
    x_mu_kal_scrambled = np.random.permutation(model_matrix['x_mu_kal'])
    x_sd_kal_scrambled = np.random.permutation(model_matrix['x_sd_kal'])
    x_sc = model_matrix['x_sc']
    subj_idx = model_matrix['subj_idx']
    y = model_matrix['y']
    n_subj = model_matrix['n_subj']

    n, d = x_mu_kal_scrambled.shape
    if sample_kwargs is None:
        sample_kwargs = dict(draws=2000, njobs=2, tune=2000, init='advi+adapt_diag')

    with pm.Model() as hier_kal_scrambeled:

        mu_1 = pm.Normal('mu_beta_kal_sc_mean', mu=0., sd=100.)
        mu_2 = pm.Normal('mu_beta_kal_sc_stdv', mu=0., sd=100.)
        mu_3 = pm.Normal('mu_beta_stick',       mu=0., sd=100.)

        sigma_1 = pm.HalfCauchy('sigma_kal_sc_means', beta=100)
        sigma_2 = pm.HalfCauchy('sigma_kal_sc_stdev', beta=100)
        sigma_3 = pm.HalfCauchy('sigma_stick',        beta=100)

        b_1 = pm.Normal('beta_kal_sc_mu',  mu=mu_1, sd=sigma_1, shape=n_subj)
        b_2 = pm.Normal('beta_kal_sc_std', mu=mu_2, sd=sigma_2, shape=n_subj)
        b_3 = pm.Normal('beta_sc',         mu=mu_3, sd=sigma_3, shape=n_subj)

        rho = \
            tt.tile(tt.reshape(b_1[subj_idx], (n, 1)), d) * x_mu_kal_scrambled + \
            tt.tile(tt.reshape(b_2[subj_idx], (n, 1)), d) * x_sd_kal_scrambled + \
            tt.tile(tt.reshape(b_3[subj_idx], (n, 1)), d) * x_sc

        p_hat = softmax(rho)

        # Data likelihood
        yl = pm.Categorical('yl', p=p_hat, observed=y)

        # inference!
        trace_kal_scram = pm.sample(**sample_kwargs)

    return hier_kal_scrambeled, trace_kal_scram


def construct_sticky_choice(raw_data, n_arms=8):
    x_sc = []
    for subj in set(raw_data['id']):
        y = pd.get_dummies(raw_data.loc[raw_data['id'] == subj, 'arm'])

        # so, not every one uses every response, so we need to correct for this
        for c in set(range(1, n_arms + 1)):
            if c not in set(y.columns):
                y[c] = np.zeros(len(y), dtype=int)
        y = y.values

        x_sc.append(np.concatenate([np.zeros((1, n_arms)), y[:-1, :]]))

    return np.concatenate(x_sc)


def construct_subj_idx(data_frame):
    subj_idx = []
    subjs_inc = {}
    for s in data_frame['id'].values:
        if s not in subjs_inc:
            subjs_inc[s] = len(subjs_inc)
        subj_idx.append(subjs_inc[s])
    subj_idx = np.array(subj_idx)
    return subj_idx


def get_loo(model, trace):
    loo = pm.stats.loo(trace, model)
    return dict(LOO=loo.LOO, LOO_se=loo.LOO_se, p_LOO=loo.p_LOO, shape_warn=loo.shape_warn)


def run_save_models(model_matrix, name_tag, sample_kwargs=None):

    sampler_args = [model_matrix, sample_kwargs]

    def sample_model(sampler, list_params, name=None):
        model, trace = sampler(*sampler_args)
        _loo = pd.DataFrame([get_loo(model, trace)], index=[name])
        _params = pm.summary(trace).loc[list_params, :]
        _params['Model'] = name
        return _loo, _params

    # GP-RBF

    rbf_params = ['mu_beta_rbf_mean', 'mu_beta_rbf_stdv', 'mu_beta_stick',
                  'sigma_rbf_means', 'sigma_rbf_stdev', 'sigma_stick']

    model_loo, model_params = sample_model(sample_heir_rbf_kal, rbf_params, 'GP-RBF')
    model_params.to_pickle('Data/model_fits/model_params_%s_rbf.pkl' % name_tag)
    model_loo.to_pickle('Data/model_fits/model_fits_%s.pkl' % name_tag)

    # Linear Regression

    lin_params = ['mu_beta_lin_mean', 'mu_beta_lin_stdv', 'mu_beta_stick',
                  'sigma_lin_means', 'sigma_lin_stdev', 'sigma_stick']
    _loo, model_params = sample_model(sample_hier_lin, lin_params, 'Lin-Reg')
    model_params.to_pickle('Data/model_fits/model_params_%s_lin.pkl' % name_tag)
    model_loo = pd.concat([model_loo, _loo])
    model_loo.to_pickle('Data/model_fits/model_fits_%s.pkl' % name_tag)

    # Clustering model

    cls_params = ['mu_beta_cls_mean','mu_beta_cls_stdv','mu_beta_stick',
                  'sigma_cls_means','sigma_cls_stdev','sigma_stick']
    _loo, model_params = sample_model(sample_hier_cls, cls_params, 'Clustering')
    model_params.to_pickle('Data/model_fits/model_params_%s_cls.pkl' % name_tag)
    model_loo = pd.concat([model_loo, _loo])
    model_loo.to_pickle('Data/model_fits/model_fits_%s.pkl' % name_tag)

    # Kalman Filter

    kal_params = ['mu_beta_kal_mean','mu_beta_kal_stdv','mu_beta_stick',
                  'sigma_kal_means','sigma_kal_stdev','sigma_stick']
    _loo, model_params = sample_model(sample_hier_cls, kal_params, 'Kalman')
    model_params.to_pickle('Data/model_fits/model_params_%s_kal.pkl' % name_tag)
    model_loo = pd.concat([model_loo, _loo])
    model_loo.to_pickle('Data/model_fits/model_fits_%s.pkl' % name_tag)

    # Bayesian GP

    bgp_params = ['mu_beta_bgp_mean','mu_beta_bgp_stdv','mu_beta_stick',
                  'sigma_bgp_means','sigma_bgp_stdev','sigma_stick']
    _loo, model_params = sample_model(sample_hier_bayes_gp, bgp_params, 'Bayesian-GP')
    model_params.to_pickle('Data/model_fits/model_params_%s_bgp.pkl' % name_tag)
    model_loo = pd.concat([model_loo, _loo])
    model_loo.to_pickle('Data/model_fits/model_fits_%s.pkl' % name_tag)

    # GP-RBF + Clustering

    rbf_cls_params = ['mu_beta_rbf_mean', 'mu_beta_rbf_stdv', 'mu_beta_cls_mean',
                      'mu_beta_cls_stdv', 'mu_beta_stick', 'sigma_rbf_means',
                      'sigma_rbf_stdev', 'sigma_cls_means','sigma_cls_stdev', 'sigma_stick']
    _loo, model_params = sample_model(sample_hier_rbf_cls, rbf_cls_params, 'GP-RBF/Clustering')
    model_params.to_pickle('Data/model_fits/model_params_%s_rbf_cls.pkl' % name_tag)
    model_loo = pd.concat([model_loo, _loo])
    model_loo.to_pickle('Data/model_fits/model_fits_%s.pkl' % name_tag)

    # Lin Reg + Clustering

    lin_cls_params = ['mu_beta_lin_mean', 'mu_beta_lin_stdv', 'mu_beta_cls_mean',
                      'mu_beta_cls_stdv', 'mu_beta_stick', 'sigma_lin_means',
                      'sigma_lin_stdev', 'sigma_cls_means', 'sigma_cls_stdev','sigma_stick']
    _loo, model_params = sample_model(sample_heir_lin_cls, lin_cls_params, 'Linear/Clustering')
    model_params.to_pickle('Data/model_fits/model_params_%s_lin_cls.pkl' % name_tag)
    model_loo = pd.concat([model_loo, _loo])
    model_loo.to_pickle('Data/model_fits/model_fits_%s.pkl' % name_tag)

    # GP-RBF + Kalman

    lin_cls_params = ['mu_beta_rbf_mean', 'mu_beta_rbf_stdv', 'mu_beta_kal_mean',
                      'mu_beta_kal_stdv', 'mu_beta_stick', 'sigma_rbf_means',
                      'sigma_rbf_stdev', 'sigma_kal_means', 'sigma_rbf_stdev', 'sigma_stick']
    _loo, model_params = sample_model(sample_heir_lin_cls, lin_cls_params, 'Linear/Clustering')
    model_params.to_pickle('Data/model_fits/model_params_%s_rbf_kal.pkl' % name_tag)
    model_loo = pd.concat([model_loo, _loo])
    model_loo.to_pickle('Data/model_fits/model_fits_%s.pkl' % name_tag)

    # Scrambled
    scram_params = ['mu_beta_kal_sc_mean', 'mu_beta_kal_sc_stdv', 'mu_beta_stick',
                      'sigma_kal_sc_means', 'sigma_kal_sc_stdev', 'sigma_stick']
    _loo, model_params = sample_model(sample_heir_scram_kal, scram_params, 'Scrambled')
    model_params.to_pickle('Data/model_fits/model_params_%s_scram.pkl' % name_tag)
    model_loo = pd.concat([model_loo, _loo])
    model_loo.to_pickle('Data/model_fits/model_fits_%s.pkl' % name_tag)


def make_debug_matrix(model_matrix, n_debug=8):
    model_matrix['x_mu_cls'] = model_matrix['x_mu_cls'][model_matrix['subj_idx'] < n_debug, :]
    model_matrix['x_sd_cls'] = model_matrix['x_sd_cls'][model_matrix['subj_idx'] < n_debug, :]
    model_matrix['x_mu_lin'] = model_matrix['x_mu_lin'][model_matrix['subj_idx'] < n_debug, :]
    model_matrix['x_sd_lin'] = model_matrix['x_sd_lin'][model_matrix['subj_idx'] < n_debug, :]
    model_matrix['x_mu_rbf'] = model_matrix['x_mu_rbf'][model_matrix['subj_idx'] < n_debug, :]
    model_matrix['x_sd_rbf'] = model_matrix['x_sd_rbf'][model_matrix['subj_idx'] < n_debug, :]
    model_matrix['x_mu_kal'] = model_matrix['x_mu_kal'][model_matrix['subj_idx'] < n_debug, :]
    model_matrix['x_sd_kal'] = model_matrix['x_sd_kal'][model_matrix['subj_idx'] < n_debug, :]
    model_matrix['x_mu_bayes_gp'] = model_matrix['x_mu_bayes_gp'][model_matrix['subj_idx'] < n_debug, :]
    model_matrix['x_sd_bayes_gp'] = model_matrix['x_sd_bayes_gp'][model_matrix['subj_idx'] < n_debug, :]
    model_matrix['y'] = model_matrix['y'][model_matrix['subj_idx'] < n_debug]
    model_matrix['x_sc'] = model_matrix['x_sc'][model_matrix['subj_idx'] < n_debug, :]
    model_matrix['subj_idx'] = model_matrix['subj_idx'][model_matrix['subj_idx'] < n_debug]
    model_matrix['n_subj'] = len(set(model_matrix['subj_idx']))

    return model_matrix


def exp_linear(sample_kwargs=None, debug=False):
    clustering_data = pd.read_pickle('Data/exp_linear/exp_lin_clustering_means_std.pkl')
    clustering_data.index = range(len(clustering_data))

    lin_gp_data = pd.read_csv('Data/exp_linear/linpred.csv')
    lin_gp_data.index = range(len(lin_gp_data))

    rbf_gp_data = pd.read_csv('Data/exp_linear/rbfpred.csv')
    rbf_gp_data.index = range(len(rbf_gp_data))

    kalman_data = pd.read_csv('Data/exp_linear/kalmanpred.csv')
    kalman_data.index = range(len(kalman_data))

    bayes_gp_data = pd.read_pickle('Data/exp_linear/bayes_gp_exp1.pkl')
    bayes_gp_data.index = range(len(bayes_gp_data))

    raw_data = pd.read_csv('Data/exp_linear/lindata.csv', header=0)
    raw_data.drop(raw_data.columns.tolist()[0], axis=1, inplace=True)

    # the GP-RBF can fail if subject always choose the same response. For simplicity, we are dropping those
    # subjects
    subjects_to_drop = set()
    for s in set(raw_data.id):
        if s not in set(rbf_gp_data.id):
            subjects_to_drop.add(s)

    for s in subjects_to_drop:
        clustering_data = clustering_data[clustering_data['Subject'] != s].copy()
        lin_gp_data = lin_gp_data[lin_gp_data.id != s].copy()
        raw_data = raw_data[raw_data.id != s].copy()
        kalman_data = kalman_data[kalman_data.id != s].copy()
        bayes_gp_data = bayes_gp_data[bayes_gp_data['Subject'] != s].copy()

    # construct a sticky choice predictor. This is the same for all of the models
    x_sc = construct_sticky_choice(raw_data)

    # PYMC3 doesn't care about the actual subject numbers, so remap these to a sequential list
    subj_idx = construct_subj_idx(lin_gp_data)
    n_subj = len(set(subj_idx))

    # prep the predictor vectors
    x_mu_cls = np.array([clustering_data.loc[:, 'mu_%d' % ii].values for ii in range(8)]).T
    x_sd_cls = np.array([clustering_data.loc[:, 'std_%d' % ii].values for ii in range(8)]).T

    x_mu_kal = np.array([kalman_data.loc[:, 'mu_%d' % ii].values for ii in range(8)]).T
    x_sd_kal = np.array([kalman_data.loc[:, 'sig_%d' % ii].values for ii in range(8)]).T

    x_mu_bayes_gp = np.array([bayes_gp_data.loc[:, 'mu_%d' % ii].values for ii in range(8)]).T
    x_sd_bayes_gp = np.array([bayes_gp_data.loc[:, 'std_%d' % ii].values for ii in range(8)]).T

    x_mu_rbf = np.array([rbf_gp_data.loc[:, 'mu_ %d' % ii].values for ii in range(8)]).T
    x_sd_rbf = np.array([rbf_gp_data.loc[:, 'sig_ %d' % ii].values for ii in range(8)]).T

    x_mu_lin = np.array([lin_gp_data.loc[:, 'mu_ %d' % ii].values for ii in range(8)]).T
    x_sd_lin = np.array([lin_gp_data.loc[:, 'sig_ %d' % ii].values for ii in range(8)]).T

    y = raw_data['arm'].values - 1  # convert to 0 indexing

    model_matrix = dict(
        x_mu_cls=x_mu_cls,
        x_sd_cls=x_sd_cls,
        x_mu_lin=x_mu_lin,
        x_sd_lin=x_sd_lin,
        x_mu_rbf=x_mu_rbf,
        x_sd_rbf=x_sd_rbf,
        x_mu_kal=x_mu_kal,
        x_sd_kal=x_sd_kal,
        x_mu_bayes_gp=x_mu_bayes_gp,
        x_sd_bayes_gp=x_sd_bayes_gp,
        y=y,
        n_subj=n_subj,
        x_sc=x_sc,
        subj_idx=subj_idx
    )

    if debug:
        model_matrix = make_debug_matrix(model_matrix)

    print "Experiment 1, Running %d subjects" % model_matrix['n_subj']
    run_save_models(model_matrix, name_tag='exp_lin', sample_kwargs=sample_kwargs)


def exp_scrambled(sample_kwargs=None, debug=False):
    clustering_data = pd.read_pickle('Data/exp_scrambled/exp_scram_clustering_means_std.pkl')
    clustering_data.index = range(len(clustering_data))

    lin_gp_data = pd.read_csv('Data/exp_scrambled/gplinscrambled.csv')
    lin_gp_data.index = range(len(lin_gp_data))

    rbf_gp_data = pd.read_csv('Data/exp_scrambled/gprbfscrambled.csv')
    rbf_gp_data.index = range(len(rbf_gp_data))

    kalman_data = pd.read_csv('Data/exp_scrambled/kalmanscrabled.csv')
    kalman_data.index = range(len(kalman_data))

    bayes_gp_data = pd.read_pickle('Data/exp_scrambled/bayes_gp_exp_scram.pkl')
    bayes_gp_data.index = range(len(bayes_gp_data))

    raw_data = pd.read_csv('Data/exp_scrambled/datascrambled.csv', header=0)
    raw_data.drop(raw_data.columns.tolist()[0], axis=1, inplace=True)

    # the GP-RBF can fail if subject always choose the same response. For simplicity, we are dropping those
    # subjects
    subjects_to_drop = set()
    for s in set(raw_data.id):
        if s not in set(rbf_gp_data.id):
            subjects_to_drop.add(s)

    for s in subjects_to_drop:
        clustering_data = clustering_data[clustering_data['Subject'] != s].copy()
        lin_gp_data = lin_gp_data[lin_gp_data.id != s].copy()
        raw_data = raw_data[raw_data.id != s].copy()
        kalman_data = kalman_data[kalman_data.id != s].copy()
        bayes_gp_data = bayes_gp_data[bayes_gp_data['Subject'] != s].copy()

    # construct a sticky choice predictor. This is the same for all of the models
    x_sc = construct_sticky_choice(raw_data)

    # PYMC3 doesn't care about the actual subject numbers, so remap these to a sequential list
    subj_idx = construct_subj_idx(lin_gp_data)
    n_subj = len(set(subj_idx))

    # prep the predictor vectors
    x_mu_cls = np.array([clustering_data.loc[:, 'mu_%d' % ii].values for ii in range(8)]).T
    x_sd_cls = np.array([clustering_data.loc[:, 'std_%d' % ii].values for ii in range(8)]).T

    x_mu_lin = np.array([lin_gp_data.loc[:, 'mu_%d' % ii].values for ii in range(8)]).T
    x_sd_lin = np.array([lin_gp_data.loc[:, 'std_%d' % ii].values for ii in range(8)]).T

    x_mu_bayes_gp = np.array([bayes_gp_data.loc[:, 'mu_%d' % ii].values for ii in range(8)]).T
    x_sd_bayes_gp = np.array([bayes_gp_data.loc[:, 'std_%d' % ii].values for ii in range(8)]).T

    x_mu_rbf = np.array([rbf_gp_data.loc[:, 'mu_%d' % ii].values for ii in range(8)]).T
    x_sd_rbf = np.array([rbf_gp_data.loc[:, 'std_%d' % ii].values for ii in range(8)]).T

    x_mu_kal = np.array([kalman_data.loc[:, 'mu_%d' % ii].values for ii in range(8)]).T
    x_sd_kal = np.array([kalman_data.loc[:, 'sig_%d' % ii].values for ii in range(8)]).T

    y = raw_data['arm'].values - 1  # convert to 0 indexing

    model_matrix = dict(
        x_mu_cls=x_mu_cls,
        x_sd_cls=x_sd_cls,
        x_mu_lin=x_mu_lin,
        x_sd_lin=x_sd_lin,
        x_mu_rbf=x_mu_rbf,
        x_sd_rbf=x_sd_rbf,
        x_mu_kal=x_mu_kal,
        x_sd_kal=x_sd_kal,
        x_mu_bayes_gp=x_mu_bayes_gp,
        x_sd_bayes_gp=x_sd_bayes_gp,
        y=y,
        n_subj=n_subj,
        x_sc=x_sc,
        subj_idx=subj_idx
    )

    if debug:
        model_matrix = make_debug_matrix(model_matrix)

    print "Experiment Scrambled, Running %d subjects" % model_matrix['n_subj']
    run_save_models(model_matrix, name_tag='exp_scram', sample_kwargs=sample_kwargs)

if __name__ == "__main__":

    exp_linear()
    exp_scrambled()
