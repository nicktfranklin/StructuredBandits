import numpy as np
import pandas as pd

import pymc3 as pm
import fit_choice_models


def comp_a(model_matrix, file_name):

    # fit the two models
    model_rbfcls, trace_rbfcls = fit_choice_models.sample_hier_rbf_cls(model_matrix)
    model_rbfkal, trace_rbfkal = fit_choice_models.sample_heir_rbf_kal(model_matrix)
    model_kal, trace_kal = fit_choice_models.sample_hier_kal(model_matrix)
    model_bgp, trace_bgp = fit_choice_models.sample_hier_bayes_gp(model_matrix)

    # compare
    df_comp_loo = pm.compare(
        {
            model_rbfcls: trace_rbfcls,
            model_rbfkal: trace_rbfkal,
            model_kal: trace_kal,
            model_bgp: trace_bgp,
        }, ic='LOO')

    df_comp_loo.rename(index={0: 'RBF/Cluster', 1: 'RBF/Kalman', 2:'Kalman', 3:'Bayesian-GP'}, inplace=True)
    df_comp_loo.to_pickle(file_name)


def comp_b(model_matrix, file_name):

    # fit the two models
    model_rbfcls, trace_rbfcls = fit_choice_models.sample_hier_rbf_cls(model_matrix)
    model_rbfkal, trace_rbfkal = fit_choice_models.sample_heir_rbf_kal(model_matrix)
    model_rbf, trace_rbf = fit_choice_models.sample_hier_rbf(model_matrix)
    model_cls, trace_cls = fit_choice_models.sample_hier_cls(model_matrix)

    # compare
    df_comp_loo = pm.compare(
        {
            model_rbfcls: trace_rbfcls,
            model_rbfkal: trace_rbfkal,
            model_rbf: trace_rbf,
            model_cls: trace_cls,
        }, ic='LOO')

    df_comp_loo.rename(index={0: 'RBF/Cluster', 1: 'RBF/Kalman', 2: 'RBF', 3: 'Clustering'}, inplace=True)
    df_comp_loo.to_pickle(file_name)


def comp_c(model_matrix, file_name):

    # fit the two models
    model_rbfkal, trace_rbfkal = fit_choice_models.sample_heir_rbf_kal(model_matrix)
    model_kal, trace_kal = fit_choice_models.sample_hier_kal(model_matrix)

    # compare
    df_comp_loo = pm.compare(
        {
            model_rbfkal: trace_rbfkal,
            model_kal: trace_kal,
        }, ic='LOO')

    df_comp_loo.rename(index={0: 'RBF/Kalman', 1: 'Kalman'}, inplace=True)
    df_comp_loo.to_pickle(file_name)



def comp_d(model_matrix, file_name):

    # fit the two models
    model_rbfcls, trace_rbfcls = fit_choice_models.sample_hier_rbf_cls(model_matrix)
    model_rbf, trace_rbf = fit_choice_models.sample_hier_rbf(model_matrix)
    model_cls, trace_cls = fit_choice_models.sample_hier_cls(model_matrix)
    model_kal, trace_kal = fit_choice_models.sample_hier_kal(model_matrix)

    # compare
    df_comp_loo = pm.compare(
        {
            model_rbfcls: trace_rbfcls,
            model_rbf: trace_rbf,
            model_cls: trace_cls,
            model_kal: trace_kal,
        }, ic='LOO')

    df_comp_loo.rename(index={0: 'RBF/Cluster', 1: 'RBF', 2: 'Clustering', 3: 'Kalman'}, inplace=True)
    df_comp_loo.to_pickle(file_name)


def comp_e(model_matrix, file_name):
    # fit the two models
    model_rbfcls, trace_rbfcls = fit_choice_models.sample_hier_rbf_cls(model_matrix)
    model_rbfkal, trace_rbfkal = fit_choice_models.sample_heir_rbf_kal(model_matrix)

    # compare
    df_comp_loo = pm.compare(
        {
            model_rbfcls: trace_rbfcls,
            model_rbfkal: trace_rbfkal,
        }, ic='LOO')

    df_comp_loo.rename(index={0: 'RBF/Cluster', 1: 'RBF/Kalman'}, inplace=True)
    df_comp_loo.to_pickle(file_name)

def comp_f(model_matrix, file_name):
    # fit the two models
    model_rbfcls, trace_rbfcls = fit_choice_models.sample_hier_rbf_cls(model_matrix)
    model_kal, trace_kal = fit_choice_models.sample_hier_kal(model_matrix)

    # compare
    df_comp_loo = pm.compare(
        {
            model_rbfcls: trace_rbfcls,
            model_kal: trace_kal,
        }, ic='LOO')

    df_comp_loo.rename(index={0: 'RBF/Cluster', 1: 'Kalman'}, inplace=True)
    df_comp_loo.to_pickle(file_name)

def comp_g(model_matrix, file_name):

    # fit the two models
    model_rbfcls, trace_rbfcls = fit_choice_models.sample_hier_rbf_cls(model_matrix)
    model_rbfkal, trace_rbfkal = fit_choice_models.sample_heir_rbf_kal(model_matrix)
    model_rbf, trace_rbf = fit_choice_models.sample_hier_rbf(model_matrix)
    model_cls, trace_cls = fit_choice_models.sample_hier_cls(model_matrix)
    model_kal, trace_kal = fit_choice_models.sample_hier_kal(model_matrix)

    # compare
    df_comp_loo = pm.compare(
        {
            model_rbfcls: trace_rbfcls,
            model_rbfkal: trace_rbfkal,
            model_rbf: trace_rbf,
            model_cls: trace_cls,
            model_kal: trace_kal,
        }, ic='LOO')

    df_comp_loo.rename(index={0: 'RBF/Cluster', 1: 'RBF/Kalman', 2: 'RBF', 3: 'Clustering', 4: 'Kalman'}, inplace=True)
    df_comp_loo.to_pickle(file_name)


def experiment_linear(debug=False):
    clustering_data = pd.read_pickle('Data/exp_linear/exp_lin_clustering_means_std.pkl')
    clustering_data.index = range(len(clustering_data))

    lin_gp_data = pd.read_csv('Data/exp_linear/linpred.csv')
    lin_gp_data.index = range(len(lin_gp_data))

    rbf_gp_data = pd.read_csv('Data/exp_linear/rbfpred.csv')
    rbf_gp_data.index = range(len(rbf_gp_data))

    kalman_data = pd.read_pickle('Data/exp_linear/kalmanpred.pkl')
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
        kalman_data = kalman_data[kalman_data.Subject != s].copy()
        bayes_gp_data = bayes_gp_data[bayes_gp_data['Subject'] != s].copy()

    # construct a sticky choice predictor. This is the same for all of the models
    x_sc = fit_choice_models.construct_sticky_choice(raw_data)

    # PYMC3 doesn't care about the actual subject numbers, so remap these to a sequential list
    subj_idx = fit_choice_models.construct_subj_idx(lin_gp_data)
    n_subj = len(set(subj_idx))

    # prep the predictor vectors
    x_mu_cls = np.array([clustering_data.loc[:, 'mu_%d' % ii].values for ii in range(8)]).T
    x_sd_cls = np.array([clustering_data.loc[:, 'std_%d' % ii].values for ii in range(8)]).T

    x_mu_kal = np.array([kalman_data.loc[:, 'mu_%d' % ii].values for ii in range(8)]).T
    x_sd_kal = np.array([kalman_data.loc[:, 'std_%d' % ii].values for ii in range(8)]).T

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
        model_matrix = fit_choice_models.make_debug_matrix(model_matrix)

    print "Experiment 1, Running %d subjects" % model_matrix['n_subj']
    comp_a(model_matrix, './data/model_compare/ModelCompare_exp_linear.pkl')


def exp_scrambled(debug=False):
    clustering_data = pd.read_pickle('Data/exp_scrambled/exp_scram_clustering_means_std.pkl')
    clustering_data.index = range(len(clustering_data))

    lin_gp_data = pd.read_csv('Data/exp_scrambled/gplinscrambled.csv')
    lin_gp_data.index = range(len(lin_gp_data))

    rbf_gp_data = pd.read_csv('Data/exp_scrambled/gprbfscrambled.csv')
    rbf_gp_data.index = range(len(rbf_gp_data))

    kalman_data = pd.read_pickle('Data/exp_scrambled/kalmanscrambled.pkl')
    kalman_data.index = range(len(kalman_data))

    bayes_gp_data = pd.read_pickle('Data/exp_scrambled/bayes_gp_exp_scram.pkl')
    bayes_gp_data.index = range(len(bayes_gp_data))

    raw_data = pd.read_csv('Data/exp_scrambled/datascrambled.csv', header=0)

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
        kalman_data = kalman_data[kalman_data.Subject != s].copy()
        bayes_gp_data = bayes_gp_data[bayes_gp_data['Subject'] != s].copy()

    # construct a sticky choice predictor. This is the same for all of the models
    x_sc = fit_choice_models.construct_sticky_choice(raw_data)

    # PYMC3 doesn't care about the actual subject numbers, so remap these to a sequential list
    subj_idx = fit_choice_models.construct_subj_idx(lin_gp_data)
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
    x_sd_kal = np.array([kalman_data.loc[:, 'std_%d' % ii].values for ii in range(8)]).T

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
        model_matrix = fit_choice_models.make_debug_matrix(model_matrix)
    # comp_b(model_matrix, './data/model_compare/ModelCompare_exp_scrambled.pkl')
    comp_c(model_matrix, './data/model_compare/ModelCompare_exp_scrambled2.pkl')
    comp_f(model_matrix, './data/model_compare/ModelCompare_exp_scrambled3.pkl')

    print "Experiment Scrambled, Running %d subjects" % model_matrix['n_subj']


def exp_shifted(sample_kwargs=None, debug=False):
    clustering_data = pd.read_pickle('Data/exp_shifted/exp_shifted_clustering_means_std.pkl')
    clustering_data.index = range(len(clustering_data))

    lin_gp_data = pd.read_csv('Data/exp_shifted/gplinshifted.csv')
    lin_gp_data.index = range(len(lin_gp_data))

    rbf_gp_data = pd.read_csv('Data/exp_shifted/gprbfshifted.csv')
    rbf_gp_data.index = range(len(rbf_gp_data))

    kalman_data = pd.read_pickle('Data/exp_shifted/kalmanshifted.pkl')
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
        kalman_data = kalman_data[kalman_data.Subject != s].copy()
        bayes_gp_data = bayes_gp_data[bayes_gp_data['Subject'] != s].copy()

    # construct a sticky choice predictor. This is the same for all of the models
    x_sc = fit_choice_models.construct_sticky_choice(raw_data)

    # PYMC3 doesn't care about the actual subject numbers, so remap these to a sequential list
    subj_idx = fit_choice_models.construct_subj_idx(lin_gp_data)
    n_subj = len(set(subj_idx))

    intercept = raw_data['int'].values

    # prep the predictor vectors
    x_mu_cls = np.array([clustering_data.loc[:, 'mu_%d' % ii].values for ii in range(8)]).T
    x_sd_cls = np.array([clustering_data.loc[:, 'std_%d' % ii].values for ii in range(8)]).T

    x_mu_bayes_gp = np.array([bayes_gp_data.loc[:, 'mu_%d' % ii].values for ii in range(8)]).T
    x_sd_bayes_gp = np.array([bayes_gp_data.loc[:, 'std_%d' % ii].values for ii in range(8)]).T

    x_mu_lin = np.array([lin_gp_data.loc[:, 'mu_%d' % ii].values + intercept for ii in range(8)]).T
    x_sd_lin = np.array([lin_gp_data.loc[:, 'std_%d' % ii].values for ii in range(8)]).T

    x_mu_rbf = np.array([rbf_gp_data.loc[:, 'mu_%d' % ii].values + intercept for ii in range(8)]).T
    x_sd_rbf = np.array([rbf_gp_data.loc[:, 'std_%d' % ii].values for ii in range(8)]).T

    x_mu_kal = np.array([kalman_data.loc[:, 'mu_%d' % ii].values for ii in range(8)]).T
    x_sd_kal = np.array([kalman_data.loc[:, 'std_%d' % ii].values for ii in range(8)]).T

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
        model_matrix = fit_choice_models.make_debug_matrix(model_matrix)

    print "Experiment Shifted, Running %d subjects" % model_matrix['n_subj']
    comp_d(model_matrix, './data/model_compare/ModelCompare_exp_shifted.pkl')
    comp_e(model_matrix, './data/model_compare/ModelCompare_exp_shifted2.pkl')


def exp_srs(debug=False):
    clustering_data = pd.read_pickle('Data/exp_srs/exp_srs_clustering_means_std.pkl')
    clustering_data.index = range(len(clustering_data))

    lin_gp_data = pd.read_csv('Data/exp_srs/gplinsrs.csv')
    lin_gp_data.index = range(len(lin_gp_data))

    rbf_gp_data = pd.read_csv('Data/exp_srs/gprbfsrs.csv')
    rbf_gp_data.index = range(len(rbf_gp_data))

    kalman_data = pd.read_pickle('Data/exp_srs/kalmansrs.pkl')
    kalman_data.index = range(len(kalman_data))

    bayes_gp_data = pd.read_pickle('Data/exp_srs/bayes_gp_exp_srs.pkl')
    bayes_gp_data.index = range(len(bayes_gp_data))

    raw_data = pd.read_csv('Data/exp_srs/datasrs.csv', header=0)

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
        kalman_data = kalman_data[kalman_data.Subject != s].copy()
        bayes_gp_data = bayes_gp_data[bayes_gp_data['Subject'] != s].copy()

    # construct a sticky choice predictor. This is the same for all of the models
    x_sc = fit_choice_models.construct_sticky_choice(raw_data)

    # PYMC3 doesn't care about the actual subject numbers, so remap these to a sequential list
    subj_idx = fit_choice_models.construct_subj_idx(lin_gp_data)
    n_subj = len(set(subj_idx))

    # prep the predictor vectors
    x_mu_cls = np.array([clustering_data.loc[:, 'mu_%d' % ii].values for ii in range(8)]).T
    x_sd_cls = np.array([clustering_data.loc[:, 'std_%d' % ii].values for ii in range(8)]).T

    x_mu_bayes_gp = np.array([bayes_gp_data.loc[:, 'mu_%d' % ii].values for ii in range(8)]).T
    x_sd_bayes_gp = np.array([bayes_gp_data.loc[:, 'std_%d' % ii].values for ii in range(8)]).T

    x_mu_lin = np.array([lin_gp_data.loc[:, 'mu%d' % ii].values for ii in range(8)]).T
    x_sd_lin = np.array([lin_gp_data.loc[:, 'sigma%d' % ii].values for ii in range(8)]).T

    x_mu_rbf = np.array([rbf_gp_data.loc[:, 'mu%d' % ii].values for ii in range(8)]).T
    x_sd_rbf = np.array([rbf_gp_data.loc[:, 'sigma%d' % ii].values for ii in range(8)]).T

    x_mu_kal = np.array([kalman_data.loc[:, 'mu_%d' % ii].values for ii in range(8)]).T
    x_sd_kal = np.array([kalman_data.loc[:, 'std_%d' % ii].values for ii in range(8)]).T

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
        model_matrix = fit_choice_models.make_debug_matrix(model_matrix)

    print "Experiment SRS, Running %d subjects" % model_matrix['n_subj']
    # comp_d(model_matrix, './data/model_compare/ModelCompare_exp_srs.pkl')
    comp_e(model_matrix, './data/model_compare/ModelCompare_exp_srs2.pkl')


def exp_change_point(sample_kwargs=None, debug=False):
    clustering_data = pd.read_pickle('Data/exp_changepoint/exp_cp_clustering_means_std.pkl')
    clustering_data.index = range(len(clustering_data))

    lin_gp_data = pd.read_csv('Data/exp_changepoint/changelinpred.csv')
    lin_gp_data.index = range(len(lin_gp_data))

    rbf_gp_data = pd.read_csv('Data/exp_changepoint/changerbfpred.csv')
    rbf_gp_data.index = range(len(rbf_gp_data))

    kalman_data = pd.read_pickle('Data/exp_changepoint/changekalmanpred.pkl')
    kalman_data.index = range(len(kalman_data))

    bayes_gp_data = pd.read_pickle('Data/exp_changepoint/bayes_gp_exp_cp.pkl')
    bayes_gp_data.index = range(len(bayes_gp_data))

    raw_data = pd.read_csv('Data/exp_changepoint/changepoint.csv', header=0)

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
        kalman_data = kalman_data[kalman_data.Subject != s].copy()
        bayes_gp_data = bayes_gp_data[bayes_gp_data['Subject'] != s].copy()

    # construct a sticky choice predictor. This is the same for all of the models
    x_sc = fit_choice_models.construct_sticky_choice(raw_data)

    # PYMC3 doesn't care about the actual subject numbers, so remap these to a sequential list
    subj_idx = fit_choice_models.construct_subj_idx(lin_gp_data)
    n_subj = len(set(subj_idx))

    # prep the predictor vectors
    x_mu_cls = np.array([clustering_data.loc[:, 'mu_%d' % ii].values for ii in range(8)]).T
    x_sd_cls = np.array([clustering_data.loc[:, 'std_%d' % ii].values for ii in range(8)]).T

    x_mu_bayes_gp = np.array([bayes_gp_data.loc[:, 'mu_%d' % ii].values for ii in range(8)]).T
    x_sd_bayes_gp = np.array([bayes_gp_data.loc[:, 'std_%d' % ii].values for ii in range(8)]).T

    x_mu_lin = np.array([lin_gp_data.loc[:, 'mu_%d' % ii].values for ii in range(8)]).T
    x_sd_lin = np.array([lin_gp_data.loc[:, 'sigma_%d' % ii].values for ii in range(8)]).T

    x_mu_rbf = np.array([rbf_gp_data.loc[:, 'mu_%d' % ii].values for ii in range(8)]).T
    x_sd_rbf = np.array([rbf_gp_data.loc[:, 'sigma_%d' % ii].values for ii in range(8)]).T

    x_mu_kal = np.array([kalman_data.loc[:, 'mu_%d' % ii].values for ii in range(8)]).T
    x_sd_kal = np.array([kalman_data.loc[:, 'std_%d' % ii].values for ii in range(8)]).T

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
        model_matrix = fit_choice_models.make_debug_matrix(model_matrix)

    print "Experiment Change Point, Running %d subjects" % model_matrix['n_subj']
    comp_g(model_matrix, './data/model_compare/ModelCompare_exp_cp.pkl')

if __name__ == "__main__":
    experiment_linear()
    exp_scrambled()
    exp_shifted()
    exp_srs()
    exp_change_point()