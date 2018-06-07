import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

from ClusteringModel.dpvi import DPVI


def objective(alpha, kwargs):
    dpvi_filter = DPVI(kwargs['k'], n_arms=kwargs['n_arms'], mu_init=kwargs['mu_prior'],
                       var_init=kwargs['var_init'], alpha=alpha)
    dpvi_filter.estimate(kwargs['list_blocks'], kwargs['list_arms'], kwargs['list_rewards'])
    return dpvi_filter.get_model_log_prob()


def main(k=5, data_path=None, output_file=None, n_arms=8, mu_prior=25.0, var_init=5.0):

    data = pd.read_csv(data_path, header=0)

    num_cores = multiprocessing.cpu_count()

    def estimate_subj(subj_n):
        # prepare a single subject's data
        subj = data.loc[data.id == subj_n, :]

        list_blocks = subj['round'].values - 1
        list_arms = subj['arm'].values - 1
        list_rewards = subj['out'].values

        kwargs = dict(list_blocks=list_blocks, list_arms=list_arms, list_rewards=list_rewards,
                      n_arms=n_arms, k=k, mu_prior=mu_prior, var_init=var_init)
        alpha = np.arange(0.1, 10, 0.25)
        f = Parallel(n_jobs=num_cores)(delayed(objective)(a, kwargs) for a in alpha)

        return alpha[np.argmax(f)]

    n_subj = len(set(data['id']))

    data_out = list()
    for ii in tqdm(set(data['id']), total=n_subj, desc='Progress'):
        alpha = estimate_subj(ii)
        data_out.append(pd.Series(alpha, index=[ii]))
        pd.concat(data_out).to_pickle(output_file)


if __name__ == "__main__":
    main(k=20, data_path='Data/exp_linear/lindata.csv', output_file='Data/alphas/exp_lin_alphas.pkl')
    main(k=20, data_path='Data/exp_changepoint/changepoint.csv', output_file='Data/alphas/exp_cp_alphas.pkl')
    main(k=20, data_path='Data/exp_shifted/datashifted.csv', output_file='Data/alphas/exp_shifted_alphas.pkl')
    main(k=20, data_path='Data/exp_scrambled/datascrambled.csv', output_file='Data/alphas/exp_scram_alphas.pkl')
    main(k=20, data_path='Data/exp_srs/datasrs.csv', output_file='Data/alphas/exp_srs_alphas.pkl')