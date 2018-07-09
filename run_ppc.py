import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import cPickle as pickle


import pymc3 as pm
from theano.tensor.nnet.nnet import softmax
from theano import tensor as tt
import theano as t

print('Runing on PyMC3 v{}'.format(pm.__version__))
print('Runing on Theano v{}'.format(t.__version__))

from choice_models import construct_sticky_choice, construct_subj_idx

def exp_linear(sample_kwargs=None):

    clustering_data = pd.read_pickle('Data/exp_linear/exp_lin_clustering_means_std.pkl')
    clustering_data.index = range(len(clustering_data))

    lin_gp_data = pd.read_csv('Data/exp_linear/linpred.csv')
    lin_gp_data.index = range(len(lin_gp_data))

    rbf_gp_data = pd.read_csv('Data/exp_linear/rbfpred.csv')
    rbf_gp_data.index = range(len(rbf_gp_data))

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

    # construct a sticky choice predictor. This is the same for all of the models
    x_sc = construct_sticky_choice(raw_data)

    # PYMC3 doesn't care about the actual subject numbers, so remap these to a sequential list
    subj_idx = construct_subj_idx(lin_gp_data)
    n_subj = len(set(subj_idx))

    # prep the predictor vectors
    x_mu_cls = np.array([clustering_data.loc[:, 'mu_%d' % ii].values for ii in range(8)]).T
    x_sd_cls = np.array([clustering_data.loc[:, 'std_%d' % ii].values for ii in range(8)]).T

    x_mu_rbf = np.array([rbf_gp_data.loc[:, 'mu_ %d' % ii].values for ii in range(8)]).T
    x_sd_rbf = np.array([rbf_gp_data.loc[:, 'sig_ %d' % ii].values for ii in range(8)]).T


    y = raw_data['arm'].values - 1  # convert to 0 indexing

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

    ppc = pm.sample_ppc(trace_gprbf_cls, samples=500, model=hier_rbf_clus)

    for ii in range(500):
        sim_draws = raw_data.copy()
        sim_draws['arm_sim'] = ppc['yl'][ii, :] + 1
        sim_draws.to_pickle('./Data/PPC/exp_linear/sim_%d.pkl' % ii)

def exp_shifted(sample_kwargs=None):

    clustering_data = pd.read_pickle('Data/exp_shifted/exp_shifted_clustering_means_std.pkl')
    clustering_data.index = range(len(clustering_data))

    lin_gp_data = pd.read_csv('Data/exp_shifted/gplinshifted.csv')
    lin_gp_data.index = range(len(lin_gp_data))

    rbf_gp_data = pd.read_csv('Data/exp_shifted/gprbfshifted.csv')
    rbf_gp_data.index = range(len(rbf_gp_data))

    kalman_data = pd.read_csv('Data/exp_shifted/kalmanshifted.csv')
    kalman_data.index = range(len(kalman_data))

    bayes_gp_data = pd.read_pickle('Data/exp_shifted/bayes_gp_exp_shifted.pkl')
    bayes_gp_data.index = range(len(bayes_gp_data))

    raw_data = pd.read_csv('Data/exp_shifted/datashifted_withoffset.csv', header=0)

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

    intercept = raw_data['int'].values

    # prep the predictor vectors
    x_mu_cls = np.array([clustering_data.loc[:, 'mu_%d' % ii].values for ii in range(8)]).T
    x_sd_cls = np.array([clustering_data.loc[:, 'std_%d' % ii].values for ii in range(8)]).T

    x_mu_lin = np.array([lin_gp_data.loc[:, 'mu_%d' % ii].values + intercept for ii in range(8)]).T
    x_sd_lin = np.array([lin_gp_data.loc[:, 'std_%d' % ii].values for ii in range(8)]).T

    x_mu_rbf = np.array([rbf_gp_data.loc[:, 'mu_%d' % ii].values + intercept for ii in range(8)]).T
    x_sd_rbf = np.array([rbf_gp_data.loc[:, 'std_%d' % ii].values for ii in range(8)]).T

    y = raw_data['arm'].values - 1  # convert to 0 indexing

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

    ppc = pm.sample_ppc(trace_gprbf_cls, samples=500, model=hier_rbf_clus)

    for ii in range(500):
        sim_draws = raw_data.copy()
        sim_draws['arm_sim'] = ppc['yl'][ii, :] + 1
        sim_draws.to_pickle('./Data/PPC/exp_shifted/sim_%d.pkl' % ii)

if __name__ == '__main__':
    exp_linear()
    exp_shifted()
