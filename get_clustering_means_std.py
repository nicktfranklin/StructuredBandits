import numpy as np
import pandas as pd
from tqdm import tqdm

from ClusteringModel.dpvi import DPVI
from ClusteringModel.hypothesis import NoiseCluster


def generate(raw_data_path, output_file_path, k=30, n_arms=8, mu_prior=25.0, var_init=5.0,
             cluster_class=NoiseCluster, kernel=None, fit_alphas_path=None, fixed_alpha=8.3,
             intercept=False):

    data = pd.read_csv(raw_data_path, header=0)
    print data.columns

    if fit_alphas_path is not None:
        alphas = pd.read_pickle(fit_alphas_path)

    def estimate_subj(subj_n, alpha):
        # prepare a single subject's data
        subj = data.loc[data.id == subj_n, :]

        list_blocks = subj['round'].values - 1
        list_arms = subj['arm'].values - 1
        list_rewards = subj['out'].values
        if intercept:
            list_rewards += subj['int'].values

        dpvi_filter = DPVI(k, n_arms=8, mu_init=mu_prior, var_init=var_init, alpha=alpha, cluster_class=cluster_class,
                           kernel=kernel)

        n_trials = len(list_blocks)
        mus = np.zeros((n_trials, n_arms), dtype=float)
        stds = np.zeros((n_trials, n_arms), dtype=float)

        nmll = np.zeros(n_trials, dtype=float)

        for t, (b, a, r) in tqdm(enumerate(zip(list_blocks, list_arms, list_rewards)), desc='Estimating',
                                 total=n_trials, leave=False):
            score = dpvi_filter.get_nmll(b, a, r)
            if (t % 10) == 0:
                block_score = score
            else:
                block_score += score

            nmll[t] = block_score

            # first, get the means and variance of the distributions
            for a0 in range(n_arms):
                mu, std = dpvi_filter.get_mean_stdev(b, a0)
                mus[t, a0] = mu
                stds[t, a0] = std
                if np.isnan(std):
                    raise Exception
            dpvi_filter.update(b, a, r)

        df = {
            'Subject': [subj_n] * n_trials,
            'Trial': range(n_trials),
            'nmll': nmll.reshape(-1)
        }

        for a0 in range(n_arms):
            df['mu_%d' % a0] = mus[:, a0]
        for a0 in range(n_arms):
            df['std_%d' % a0] = stds[:, a0]

        return pd.DataFrame(df)

    means_std = []

    for subj_n in tqdm(set(data['id']), total=len(set(data['id']))):
        if fit_alphas_path is not None:
            alpha = alphas[subj_n]
        else:
            alpha = fixed_alpha
        means_std.append(estimate_subj(subj_n, alpha=alpha))

    pd.concat(means_std).to_pickle(output_file_path)


if __name__ == "__main__":

    #### Set parameters for all of the experiments!

    k = 20

    # linear
    generate(
        raw_data_path    = 'Data/exp_linear/lindata.csv',
        fit_alphas_path  = 'Data/alphas/exp_lin_alphas.pkl',
        output_file_path = 'Data/exp_linear/exp_lin_clustering_means_std.csv',
        k=k
    )

    # scrambled
    generate(
        raw_data_path    = 'Data/exp_scrambled/datascrambled.csv',
        fit_alphas_path  = 'Data/alphas/exp_lin_alphas.pkl',
        output_file_path = 'Data/exp_scrambled/exp_lin_clustering_means_std.csv',
        k=k
    )

    # shifted
    generate(
        raw_data_path    = 'Data/exp_shifted/datashifted.csv',
        fit_alphas_path  = 'Data/alphas/exp_shifted_alphas.pkl',
        output_file_path = 'Data/exp_shifted/exp_shifted_clustering_means_std.csv',
        k=k,
        intercept=True
    )

    # SRS
    generate(
        raw_data_path    = 'Data/exp_srs/datasrs.csv',
        fit_alphas_path  = 'Data/alphas/exp_srs_alphas.pkl',
        output_file_path = 'Data/exp_srs/exp_srs_clustering_means_std.csv',
        k=k
    )

    # changepoint
    generate(
        raw_data_path    = 'Data/exp_changepoint/changepoint.csv',
        fit_alphas_path  = 'Data/alphas/exp_cp_alphas.pkl',
        output_file_path = 'Data/exp_changepoint/exp_cp_clustering_means_std.csv',
        k=k
    )






