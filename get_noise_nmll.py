import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm


def get_noise_nmll(raw_data_path, n_arms=8, mu_prior=25.0, var_init=5.0, var_e=0.1, var_i=5., intercept=False):

    data = pd.read_csv(raw_data_path, header=0)

    def estimate_subj(subj_n):
        # prepare a single subject's data
        subj = data.loc[data.id == subj_n, :]

        list_blocks = subj['round'].values - 1
        list_arms = subj['arm'].values - 1
        list_rewards = subj['out'].values
        if intercept:
            list_rewards += subj['int'].values

        n_trials = len(list_blocks)
        mus = np.zeros((n_trials, n_arms), dtype=float) + mu_prior
        vars = np.zeros((n_trials, n_arms), dtype=float) + var_init
        nlml = np.ones((n_trials), dtype=float)

        b_old = -1

        for t, (b, a, r) in enumerate(zip(list_blocks, list_arms, list_rewards)):
            if b_old != b:
                b_old = b
                block_nlml = 1.0
            else:
                block_nlml -= norm(loc=mus[t, a], scale=vars[t, a]).logpdf(r)
            nlml[t] = block_nlml

            # update the estimates for the next trial only if not the end of a block
            if ((t + 1) % 10) > 0:

                # carry forward the previous variances and means --> these are overwritten for updated values
                mus[t+1, :] = mus[t, :]
                vars[t+1, :] = vars[t, :] + var_i  # this is the correct update!

                r0 = list_rewards[t]
                # calculate the Kalman Gain
                G = (vars[t, a] + var_i) / (vars[t, a] + var_e + var_i)

                mus[t+1, a] = mus[t, a] + G * (r0 - mus[t, a])
                # print G, (1 - G), (vars[t, a] + var_i), (1 - G) * (vars[t, a] + var_i)

                vars[t+1, a] = vars[t, a] + (1- G) * var_i  # this is the update Eric's code used (but it's wrong!)
                vars[t + 1, a] =  + (1 - G) * (vars[t, a] + var_i)

        df = {
            'Subject': [subj_n] * n_trials,
            'Trial': range(n_trials),
            'nlml': nlml
        }
        for a0 in range(n_arms):
            df['mu_%d' % a0] = mus[:, a0]
        for a0 in range(n_arms):
            df['std_%d' % a0] = vars[:, a0] ** 0.5
        return pd.DataFrame(df)

    means_std = []
    for subj_n in tqdm(set(data['id'])):
        means_std.append(estimate_subj(subj_n))
    return pd.concat(means_std)


if __name__ == "__main__":
    pass
    ## example usage!
    # out = get_noise_nmll(raw_data_path='Data/changepoint.csv')
    # out.to_pickle(output_file_path='Data/exp2_noise_mnll.pkl')
    #
    # out = get_noise_nmll(raw_data_path='Data/Shifted/datashifted.csv')
    # out.to_pickle(output_file_path='Data/Shifted/expShift_noise_mnll.pkl')
