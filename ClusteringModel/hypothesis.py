import numpy as np
from scipy.stats import norm
import GPy


class Cluster(object):

    def __init__(self, n_arms=8, mu_init=0.0, var_init=1.0):
        self.history = list()
        self.n_arms = n_arms

        # store for MAP estimates of the means
        self.mu_init = mu_init
        self.var_init = var_init

    def update(self, a, r, messages=False):
        self.history.append((a, r))
        # self.update_estimates(messages)

    def update_estimates(self, messages=False):
        pass

    def get_obs_log_prob(self, a, r):
        return 1.0

    def deepcopy(self):
        copy = Cluster(self.n_arms)
        copy.history = [(a, r) for a, r in self.history]  # this is functionally a deep copy
        # operation
        copy.update_estimates()
        return copy

    def get_mean(self, a):
        return self.mu_init

    def get_var(self, a):
        return self.var_init

    def get_mean_var(self, a):
        return self.get_mean(a), self.get_var(a)


class NoiseCluster(Cluster):
    """ each hypothesis cluster is defined by n independent bandits
    with the rewards in each normally distributed"""

    def __init__(self, n_arms=8, mu_init=0.0, var_init=1.0):
        Cluster.__init__(self, n_arms, mu_init, var_init)

        # these are the updated estimates
        self.mus = np.zeros(n_arms) + mu_init
        self.var = var_init

    def update(self, a, r, messages=False):
        self.history.append((a, r))
        self.update_estimates(messages)

    def update_estimates(self, messages=None):
        # this is a maximum likelihood estimate
        arm_tot_rewards = np.zeros(self.mus.shape)
        arm_counts = np.zeros(self.mus.shape)
        for a, r in self.history:
            arm_tot_rewards[a] += r
            arm_counts[a] += 1

        # adjust the mus for the prior (with a weight of 1 observation)
        arm_tot_rewards += self.mu_init
        arm_counts += 1

        self.mus = arm_tot_rewards / arm_counts

        # because there is prior, we can estimate the variance with as little as 1 observation
        # note! this could potentially be problematic, causing a very low variance estimate for
        # new clusters. However, I think it *should* be okay if we are updating multiple candidate
        # clusters in the background (it will wash out?).
        n = len(self.history)

        # estimate the variance as the square of the sample standard deviation
        if n > 1:
            self.var = np.sum([(self.mus[a] - r) ** 2 for a, r in self.history]) / (n - 1)
        else:
            self.var = self.var_init

    def get_obs_log_prob(self, a, r):
        return norm.logpdf(r, loc=self.mus[a], scale=np.sqrt(self.var))

    def get_mean(self, a):
        return self.mus[a]

    def get_mean_var(self, a):
        return self.get_mean(a), self.get_var(a)

    def deepcopy(self):
        """ Returns a deep copy of the cluster. Used when augmenting hypotheses """
        copy = NoiseCluster(n_arms=len(self.mus), mu_init=self.mu_init, var_init=self.var)
        copy.history = [(a, r) for a, r in self.history]  # this is functionally a deep copy
        # operation
        copy.update_estimates()
        return copy


class GPCluster(Cluster):
    def __init__(self, n_arms=8, mu_init=0.0, var_init=1.0, kernel=None):
        Cluster.__init__(self, n_arms, mu_init, var_init)
        self.X = np.zeros((1, 0))
        self.y = np.zeros(0)
        self.inv_A = np.eye(1)
        self.m = None
        self.y_offset = 0.0
        if kernel is not None:
            self.kernel = kernel
        else:
            self.kernel = GPy.kern.Linear
        self.deepcopy_kwargs = dict(n_arms=n_arms, mu_init=mu_init, var_init=var_init, kernel=self.kernel)

    def update(self, a, r, messages=False):
        self.history.append((a, r))
        self.update_estimates(messages)

    def update_estimates(self, messages=False):
        n_obs = len(self.history)
        if n_obs > 0:
            X = np.reshape([a for a, _ in self.history], (n_obs, 1))
            y = np.reshape([r for _, r in self.history], (n_obs, 1))
            self.y_offset = np.mean(y)
            self.y_offset = np.mean(y)
            if type(self.kernel) == list:
                k = None
                for k0 in self.kernel:
                    if k is None:
                        k = k0(input_dim=1)
                    else:
                        k += k0(input_dim=1)
            else:
                k = self.kernel(input_dim=1)

            self.m = GPy.models.GPRegression(X, y-self.y_offset, k)
            self.m.optimize(messages=messages)

    def get_obs_log_prob(self, a, r):
        if self.m is not None:
            return self.m.log_predictive_density(np.array([[a]]), np.array([[r-self.y_offset]]))[0][0]

    def get_model_prob(self):
        return self.m.log_likelihood()

    def get_mean_var(self, a):
        if self.m is None:
            return self.mu_init, self.var_init
        mu, var = self.m.predict(np.array([[a]]))
        return mu+self.y_offset, var

    def get_mean(self, a):
        mu, _ = self.get_mean_var(a)
        return mu+self.y_offset

    def get_var(self, a):
        _, var = self.get_mean_var(a)
        return var

    def deepcopy(self):
        """ Returns a deep copy of the cluster. Used when augmenting hypotheses """
        copy = GPCluster(**self.deepcopy_kwargs)
        copy.history = [(a, r) for a, r in self.history]  # this is functionally a deep copy
        # operation
        copy.update_estimates()
        return copy


def label_switch(assignments):
    """ utility function that ensures the 0th block is always in cluster 0, the 1st
    block is always in [0, 1], etc.  Prevents a label switching problem"""
    set_k = set()
    label_key = dict()
    for k in assignments:
        if k not in set_k:
            label_key[k] = len(set_k)
            set_k.add(k)
    label_switched_assignments = [label_key[k] for k in assignments]
    return label_switched_assignments


class Hypothesis(object):
    """docstring for Hypothesis"""

    def __init__(self, n_arms=8, mu_init=0.0, var_init=1.0, alpha=1.0, cluster_class=NoiseCluster, kernel=None):
        """
        Parameters
        ----------

        n_arms: int (default 8)
            number of bandits in a block

        mu_init: float (default 0.)
            prior over the mean of the bandit arms

        var_init: float (default 1.)
            initial value for the sigma of the arm

        alpha: float (default 1.)
            concentration parameter of the prior

        cluster_class: Cluster object (default NoiseCluster)
            internal function to estimate value over arms

        kernel: GPy kernel (None)
            only valid with GPClusters, defaults to GPy.kern.linear for GPClusters

        """

        self.cluster_assignments = list()
        self.hypotheis_kwargs = dict(n_arms=n_arms, mu_init=mu_init, var_init=var_init, alpha=alpha,
                                     cluster_class=cluster_class, kernel=kernel)
        self.alpha = alpha

        self.mu_init = mu_init
        self.var_init = var_init

        # initialize the clusters
        self.clusters = dict()

        # initialize the log prior probability
        self.log_prior = 0.0

        # initialize the log likelihood
        self.log_likelihood = 0.0

        # create a list of experiences for posterior calculations
        self.experiences = list()

        # when we create a cluster, use this class
        self.cluster_class = cluster_class
        self.cluster_kwargs = dict(n_arms=n_arms, mu_init=mu_init, var_init=var_init)
        if kernel is not None:
            self.cluster_kwargs['kernel'] = kernel

    def update(self, block, arm, reward):
        k = self.cluster_assignments[block]
        cluster = self.clusters[k]
        cluster.update(arm, reward)
        self.clusters[k] = cluster

        # Cache the experience for later
        self.experiences.append(tuple([block, arm, reward]))

    def get_mean(self, block, arm):
        """
        :param block:
        :param arm:
        :return: mu
        """
        # special case new blocks by integrating over the prior
        if block == len(self.cluster_assignments):
            # get the prior probability weights
            w = [np.sum(np.array(self.cluster_assignments) == ii) for ii in set(self.cluster_assignments)]
            w.append(self.alpha)
            w /= np.sum(w)  # normalize the prior
            mu = w[-1] * self.mu_init

            for k in range(len(w) - 1):
                cluster = self.clusters[k]
                mu += w[k] * cluster.get_mean(arm)

            return mu

        cluster = self.clusters[self.cluster_assignments[block]]
        return cluster.get_mean(arm)

    def get_var_prior(self, block, arm):
        """ this is for the special case that is a new block that has not been seen. Function returns
        the terms needed to calculate the variance over the full distribution."""
        # get the prior probability weights
        w = [np.sum(np.array(self.cluster_assignments) == ii) for ii in set(self.cluster_assignments)]
        w.append(self.alpha)
        w /= np.sum(w)  # normalize the prior

        # account for the possibility of a new cluster
        var0 = w[-1] * self.var_init
        var1 = w[-1] * (self.mu_init ** 2)
        var2 = w[-1] * self.mu_init

        for ii, cluster in enumerate(self.clusters.itervalues()):
            # the variance calculation:
            # var(f) = sum(w*sigma^2) + sum(w*mu^2) - (sum(w*mu))^2
            mu_a, var = cluster.get_mean_var(arm)
            var0 += w[ii] * cluster.get_var(arm)
            var1 += w[ii] * (mu_a ** 2)
            var2 += w[ii] * mu_a

        return var0 + var1 - (var2 ** 2)

    def get_var(self, block, a):
        cluster = self.clusters[self.cluster_assignments[block]]
        return cluster.get_var(a)

    def update_log_likelihood(self):

        self.log_likelihood = 0.0
        for b, arm, rew in self.experiences:
            cluster = self.clusters[self.cluster_assignments[b]]
            self.log_likelihood += cluster.get_obs_log_prob(arm, rew)

    def get_log_post(self):
        return self.log_likelihood + self.log_prior

    def get_obs_logprob(self, block, arm, r):
        if block < len(self.cluster_assignments):
            k = self.cluster_assignments[block]
            cluster = self.clusters[k]
            return cluster.get_obs_log_prob(arm, r)
        else:
            return norm.logpdf(r, loc=self.mu_init, scale=np.sqrt(self.var_init))

    def update_crp_prior(self):
        """ use the chinese restaurant process to calculate the prior probability of a given
        set of assignments"""

        if len(self.cluster_assignments) > 0:
            k = max(self.cluster_assignments) + 1
        else:
            k = 1

        n_k = np.zeros(k)
        self.log_prior = 0.0
        # b/c the CRP is exchangeable, it's easiest to just run through the process
        for k0 in self.cluster_assignments:
            if n_k[k0] == 0:
                # log prob of a new cluster
                self.log_prior += np.log(self.alpha / (np.sum(n_k) + self.alpha))
            else:
                # log prob of cluster reuse
                self.log_prior += np.log(n_k[k0] / (np.sum(n_k) + self.alpha))
            n_k[k0] += 1

    def update_posterior(self):
        self.update_log_likelihood()
        self.update_crp_prior()

    def augment_assignment(self, block, k):
        """
        Parameters
        ----------

        block: int
            the block index

        k: int
            the cluster index

        """

        # only augment new blocks, which are sequential for this data!
        if block < len(self.cluster_assignments):
            # return if this is not a new block of trials
            return

        # check if cluster "k" has already been assigned
        if k not in self.cluster_assignments:
            # if not, add a new reward cluster
            self.clusters[k] = self.cluster_class(**self.cluster_kwargs)

        self.cluster_assignments.append(k)
        self.cluster_assignments = label_switch(self.cluster_assignments)  # this is a check
        self.update_crp_prior()

    def deepcopy(self):
        h_copy = Hypothesis(**self.hypotheis_kwargs)
        h_copy.cluster_assignments = [k for k in self.cluster_assignments]
        h_copy.clusters = {k: cluster.deepcopy() for k, cluster in self.clusters.iteritems()}
        h_copy.experiences = [(b, a, r) for b, a, r in self.experiences]
        h_copy.log_prior = self.log_prior

        return h_copy

    # make hypotheses hashable
    def __hash__(self):
        # define the hash key as a function of the
        hash_key = ''
        for k in self.cluster_assignments:
            hash_key += str(k)
        return hash(hash_key)

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()
