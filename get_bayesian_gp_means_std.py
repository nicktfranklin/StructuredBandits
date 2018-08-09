import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.special import logsumexp
from scipy.stats import norm
from get_kalman import get_noise_nmll


def get_means_stdev(x_mu_rbf, x_sd_rbf, x_mu_lin, x_sd_lin, x_mu_kal, x_sd_kal,
                    rewards, lin_gp_data, rbf_gp_data, noise_nmll):

    log_posterior = np.zeros((300, 3))

    all_subj = []
    for s in tqdm(set(noise_nmll.Subject)):
        lin_gp_data_nll0 = lin_gp_data.loc[lin_gp_data.id == s, 'nlml'].values
        rbf_gp_data_nll0 = rbf_gp_data.loc[rbf_gp_data.id == s, 'nlml'].values
        noise_nll0 = noise_nmll.loc[noise_nmll.Subject == s, 'nlml'].values

        # each of the loss scores includes an initialization of 1.0 for the first observation,
        # and as loss is cumulative within a round, this iniatiliaztion needs to be removed from
        # all of the loss values
        lin_gp_data_nll0 -= 1.0
        rbf_gp_data_nll0 -= 1.0
        noise_nll0 -= 1.0

        # loop through the trials
        log_prior = np.zeros(3)
        for t in range(len(noise_nll0)):

            if (t % 10) == 0:
                # for each of the GP model, we need to use an initialization over the model
                # as it's prior. In the code Eric wrote, the log loss is set to -1.0 for
                # the first observation, and it will vary depending on the prior
                gp_loss = - norm(loc=25.0, scale=np.sqrt(5)).logpdf(rewards[t])

                # this loss value needs to be added to each of the observations in the round,
                # because the loss scores are cumulative within a round

            if ((t % 10) == 0) & (t > 0):
                # update the prior with the normalize posterior
                log_prior[:] = log_posterior[t-1, :] - logsumexp(log_posterior[t-1, :])

            log_posterior[t, :] = log_prior - np.array([
                noise_nll0[t]+gp_loss, rbf_gp_data_nll0[t]+gp_loss, lin_gp_data_nll0[t]+gp_loss,
            ])

        # now that we have the log posterior, use this to weight the means and stdevs
        log_posterior -= np.tile(logsumexp(log_posterior, axis=1).reshape(-1, 1), (1, 3))
        w = np.exp(log_posterior)

        # subselect the trials
        idx = np.arange(len(noise_nmll))[np.array(noise_nmll.Subject == s)]
        w_kal = np.tile(np.array(w[:, 0]).reshape(-1, 1), (1, 8))
        w_rbf = np.tile(np.array(w[:, 1]).reshape(-1, 1), (1, 8))
        w_lin = np.tile(np.array(w[:, 2]).reshape(-1, 1), (1, 8))
        mu_mix = x_mu_kal[idx, :] * w_kal + x_mu_rbf[idx, :] * w_rbf + x_mu_lin[idx, :] * w_lin

        var0 = w_kal * (x_sd_kal[idx, :] ** 2) + \
               w_rbf * (x_sd_rbf[idx, :] ** 2) + \
               w_lin * (x_sd_lin[idx, :] ** 2)

        var1 = w_kal * (x_mu_kal[idx, :] ** 2) + \
               w_rbf * (x_mu_rbf[idx, :] ** 2) + \
               w_lin * (x_mu_lin[idx, :] ** 2)

        var2 = w_kal * x_mu_kal[idx, :] + \
               w_rbf * x_mu_rbf[idx, :] + \
               w_lin * x_mu_lin[idx, :]

        std_mix = np.sqrt(var0 + var1 - (var2 ** 2))

        subj_df = {
            'Subject': [s] * len(mu_mix),
            'Trial': range(len(mu_mix)),
            'log p(Noise)': log_posterior[:, 0],
            'log p(RBF)': log_posterior[:, 1],
            'log p(Lin)': log_posterior[:, 2],
        }

        # calculate a weighted loss function
        nmll = logsumexp([
            noise_nll0 + log_posterior[:, 0],
            rbf_gp_data_nll0 + log_posterior[:, 1],
            lin_gp_data_nll0 + log_posterior[:, 2],
            ], axis=0)

        subj_df['nmll'] = nmll

        for a0 in range(np.shape(mu_mix)[1]):
            subj_df['mu_%d' % a0] = mu_mix[:, a0]
            subj_df['std_%d' % a0] = std_mix[:, a0]

        all_subj.append(pd.DataFrame(subj_df))

    return pd.concat(all_subj)

# N.B. each experiment needs a seperate function to prepare it's own data

def exp_lin():
    lin_gp_data = pd.read_csv('Data/exp_linear/linpred.csv')
    lin_gp_data.index = range(len(lin_gp_data))

    rbf_gp_data = pd.read_csv('Data/exp_linear/rbfpred.csv')
    rbf_gp_data.index = range(len(rbf_gp_data))

    raw_data = pd.read_csv('Data/exp_linear/lindata.csv')
    rewards = raw_data['out'].values

    noise_nmll = get_noise_nmll(raw_data_path='Data/exp_linear/lindata.csv')

    # drop subjects for which the RBF failed to converge
    subjects_to_drop = set()
    for s in set(noise_nmll.Subject):
        if s not in set(rbf_gp_data.id):
            subjects_to_drop.add(s)

    for s in subjects_to_drop:
        lin_gp_data = lin_gp_data[lin_gp_data.id != s].copy()
        noise_nmll = noise_nmll[noise_nmll.Subject != s].copy()

    x_mu_rbf = np.array([rbf_gp_data.loc[:, 'mu_ %d' % ii].values for ii in range(8)]).T
    x_sd_rbf = np.array([rbf_gp_data.loc[:, 'sig_ %d' % ii].values for ii in range(8)]).T

    x_mu_lin = np.array([lin_gp_data.loc[:, 'mu_ %d' % ii].values for ii in range(8)]).T
    x_sd_lin = np.array([lin_gp_data.loc[:, 'sig_ %d' % ii].values for ii in range(8)]).T

    x_mu_kal = np.array([noise_nmll.loc[:, 'mu_%d' % ii].values for ii in range(8)]).T
    x_sd_kal = np.array([noise_nmll.loc[:, 'std_%d' % ii].values for ii in range(8)]).T

    # get the posterior over models for each subject across time
    # print lin_gp_data.columns
    all_subjs = get_means_stdev(
        x_mu_rbf, x_sd_rbf, x_mu_lin, x_sd_lin, x_mu_kal, x_sd_kal,
        rewards, lin_gp_data, rbf_gp_data, noise_nmll
        )
    all_subjs.to_pickle('Data/exp_linear/bayes_gp_exp1.pkl')


def exp_shifted():

    lin_gp_data = pd.read_csv('Data/exp_shifted/gplinshifted.csv')
    lin_gp_data.index = range(len(lin_gp_data))

    rbf_gp_data = pd.read_csv('Data/exp_shifted/gprbfshifted.csv')
    rbf_gp_data.index = range(len(rbf_gp_data))

    raw_data = pd.read_csv('Data/exp_shifted/datashifted_withoffset.csv')
    rewards = raw_data['out'].values + raw_data['int'].values

    noise_nmll = get_noise_nmll(raw_data_path='Data/exp_shifted/datashifted_withoffset.csv', intercept=True)

    # drop subjects for which the RBF failed to converge
    subjects_to_drop = set()
    for s in set(noise_nmll.Subject):
        if s not in set(rbf_gp_data.id):
            subjects_to_drop.add(s)

    for s in subjects_to_drop:
        lin_gp_data = lin_gp_data[lin_gp_data.id != s].copy()
        noise_nmll = noise_nmll[noise_nmll.Subject != s].copy()

    x_mu_lin = np.array([lin_gp_data.loc[:, 'mu_%d' % ii].values + raw_data['int'].values for ii in range(8)]).T
    x_sd_lin = np.array([lin_gp_data.loc[:, 'std_%d' % ii].values for ii in range(8)]).T

    x_mu_rbf = np.array([rbf_gp_data.loc[:, 'mu_%d' % ii].values + raw_data['int'].values for ii in range(8)]).T
    x_sd_rbf = np.array([rbf_gp_data.loc[:, 'std_%d' % ii].values for ii in range(8)]).T

    x_mu_kal = np.array([noise_nmll.loc[:, 'mu_%d' % ii].values for ii in range(8)]).T
    x_sd_kal = np.array([noise_nmll.loc[:, 'std_%d' % ii].values for ii in range(8)]).T

    all_subjs = get_means_stdev(
        x_mu_rbf, x_sd_rbf, x_mu_lin, x_sd_lin, x_mu_kal, x_sd_kal,
        rewards, lin_gp_data, rbf_gp_data, noise_nmll
        )
    all_subjs.to_pickle('Data/exp_shifted/bayes_gp_exp_shifted.pkl')


def exp_cp():

    lin_gp_data = pd.read_csv('Data/exp_changepoint/changelinpred.csv')
    lin_gp_data.index = range(len(lin_gp_data))

    rbf_gp_data = pd.read_csv('Data/exp_changepoint/changerbfpred.csv')
    rbf_gp_data.index = range(len(rbf_gp_data))

    raw_data = pd.read_csv('Data/exp_changepoint/changepoint.csv')
    rewards = raw_data['out'].values

    noise_nmll = get_noise_nmll(raw_data_path='Data/exp_changepoint/changepoint.csv')

    # drop subjects for which the RBF failed to converge
    subjects_to_drop = set()
    for s in set(noise_nmll.Subject):
        if s not in set(rbf_gp_data.id):
            subjects_to_drop.add(s)

    for s in subjects_to_drop:
        lin_gp_data = lin_gp_data[lin_gp_data.id != s].copy()
        noise_nmll = noise_nmll[noise_nmll.Subject != s].copy()

    x_mu_lin = np.array([lin_gp_data.loc[:, 'mu_%d' % ii].values for ii in range(8)]).T
    x_sd_lin = np.array([lin_gp_data.loc[:, 'sigma_%d' % ii].values for ii in range(8)]).T

    x_mu_rbf = np.array([rbf_gp_data.loc[:, 'mu_%d' % ii].values for ii in range(8)]).T
    x_sd_rbf = np.array([rbf_gp_data.loc[:, 'sigma_%d' % ii].values for ii in range(8)]).T

    x_mu_kal = np.array([noise_nmll.loc[:, 'mu_%d' % ii].values for ii in range(8)]).T
    x_sd_kal = np.array([noise_nmll.loc[:, 'std_%d' % ii].values for ii in range(8)]).T

    all_subjs = get_means_stdev(
        x_mu_rbf, x_sd_rbf, x_mu_lin, x_sd_lin, x_mu_kal, x_sd_kal,
        rewards, lin_gp_data, rbf_gp_data, noise_nmll
        )
    all_subjs.to_pickle('Data/exp_changepoint/bayes_gp_exp_cp.pkl')


def exp_srs():

    lin_gp_data = pd.read_csv('Data/exp_srs/gplinsrs.csv')
    lin_gp_data.index = range(len(lin_gp_data))

    rbf_gp_data = pd.read_csv('Data/exp_srs/gprbfsrs.csv')
    rbf_gp_data.index = range(len(rbf_gp_data))

    raw_data = pd.read_csv('Data/exp_srs/datasrs.csv')
    rewards = raw_data['out'].values

    noise_nmll = get_noise_nmll(raw_data_path='Data/exp_srs/datasrs.csv')

    # drop subjects for which the RBF failed to converge
    subjects_to_drop = set()
    for s in set(noise_nmll.Subject):
        if s not in set(rbf_gp_data.id):
            subjects_to_drop.add(s)

    for s in subjects_to_drop:
        lin_gp_data = lin_gp_data[lin_gp_data.id != s].copy()
        noise_nmll = noise_nmll[noise_nmll.Subject != s].copy()

    x_mu_lin = np.array([lin_gp_data.loc[:, 'mu%d' % ii].values for ii in range(8)]).T
    x_sd_lin = np.array([lin_gp_data.loc[:, 'sigma%d' % ii].values for ii in range(8)]).T

    x_mu_rbf = np.array([rbf_gp_data.loc[:, 'mu%d' % ii].values for ii in range(8)]).T
    x_sd_rbf = np.array([rbf_gp_data.loc[:, 'sigma%d' % ii].values for ii in range(8)]).T

    x_mu_kal = np.array([noise_nmll.loc[:, 'mu_%d' % ii].values for ii in range(8)]).T
    x_sd_kal = np.array([noise_nmll.loc[:, 'std_%d' % ii].values for ii in range(8)]).T

    all_subjs = get_means_stdev(
        x_mu_rbf, x_sd_rbf, x_mu_lin, x_sd_lin, x_mu_kal, x_sd_kal,
        rewards, lin_gp_data, rbf_gp_data, noise_nmll
        )
    all_subjs.to_pickle('Data/exp_srs/bayes_gp_exp_srs.pkl')

def exp_scrambled():

    lin_gp_data = pd.read_csv('Data/exp_scrambled/gplinscrambled.csv')
    lin_gp_data.index = range(len(lin_gp_data))

    rbf_gp_data = pd.read_csv('Data/exp_scrambled/gprbfscrambled.csv')
    rbf_gp_data.index = range(len(rbf_gp_data))

    raw_data = pd.read_csv('Data/exp_scrambled/datascrambled.csv')
    rewards = raw_data['out'].values

    noise_nmll = get_noise_nmll(raw_data_path='Data/exp_scrambled/datascrambled.csv')

    # drop subjects for which the RBF failed to converge
    subjects_to_drop = set()
    for s in set(noise_nmll.Subject):
        if s not in set(rbf_gp_data.id):
            subjects_to_drop.add(s)

    for s in subjects_to_drop:
        lin_gp_data = lin_gp_data[lin_gp_data.id != s].copy()
        noise_nmll = noise_nmll[noise_nmll.Subject != s].copy()

    x_mu_lin = np.array([lin_gp_data.loc[:, 'mu_%d' % ii].values for ii in range(8)]).T
    x_sd_lin = np.array([lin_gp_data.loc[:, 'std_%d' % ii].values for ii in range(8)]).T

    x_mu_rbf = np.array([rbf_gp_data.loc[:, 'mu_%d' % ii].values for ii in range(8)]).T
    x_sd_rbf = np.array([rbf_gp_data.loc[:, 'std_%d' % ii].values for ii in range(8)]).T

    x_mu_kal = np.array([noise_nmll.loc[:, 'mu_%d' % ii].values for ii in range(8)]).T
    x_sd_kal = np.array([noise_nmll.loc[:, 'std_%d' % ii].values for ii in range(8)]).T

    all_subjs = get_means_stdev(
        x_mu_rbf, x_sd_rbf, x_mu_lin, x_sd_lin, x_mu_kal, x_sd_kal,
        rewards, lin_gp_data, rbf_gp_data, noise_nmll
        )
    all_subjs.to_pickle('Data/exp_scrambled/bayes_gp_exp_scram.pkl')


if __name__ == "__main__":
    exp_lin()
    exp_shifted()
    exp_cp()
    exp_srs()
    exp_scrambled()
