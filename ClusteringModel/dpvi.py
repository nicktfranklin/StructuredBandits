import numpy as np
from scipy.misc import logsumexp

from ClusteringModel.hypothesis import Hypothesis, NoiseCluster


def augment_assignments(cluster_assignments, new_context):
    if (len(cluster_assignments) == 0) | (len(cluster_assignments[0]) == 0):
        _cluster_assignments = list()
        _cluster_assignments.append({new_context: 0})
    else:
        _cluster_assignments = list()
        for assignment in cluster_assignments:
            new_list = list()
            for k in range(0, max(assignment.values()) + 2):
                _assignment_copy = assignment.copy()
                _assignment_copy[new_context] = k
                new_list.append(_assignment_copy)
            _cluster_assignments += new_list

    return _cluster_assignments


def enumerate_assignments(max_context_number):
    """
     enumerate all possible assignments of contexts to clusters for a fixed number of contexts.
     Has the hard assumption that the first context belongs to cluster #1, to remove redundant
     assignments that differ in labeling.

    :param max_context_number: int
    :return:     list of lists, each a function that takes in a context id number and returns a
                cluster id number
    """
    cluster_assignments = [{}]  # context 0 is always in cluster 1

    for contextNumber in range(0, max_context_number):
        cluster_assignments = augment_assignments(cluster_assignments, contextNumber)
    return cluster_assignments

def count_hypothesis_space(n_contexts):
    """
     Determine the number of unique hypotheses in the clustering space
     """
    return len(enumerate_assignments(n_contexts))


class DPVI(object):
    """docstring for Discrete particle variational inference"""
    def __init__(self, k, n_arms=8, mu_init=0., var_init=1., alpha=1.0, cluster_class=NoiseCluster, kernel=None):
        """
        Parameters
        ----------

        k:    int
            number of particles

        mu_init:    float (default 0.0)
            prior for mu

        var_init: float (default 1.0)
            initial value of sigma

        alpha: float (default 1.0)
            concentration parameter for the CRP
        """
        self.k = k
        self.hypothesis_kwargs = dict(n_arms=n_arms, alpha=alpha, mu_init=mu_init, var_init=var_init,
                                      cluster_class=cluster_class, kernel=kernel)

        # initialize the k hypotheses (really, there is only 1 hypothesis here but it is up to k hypotheses)
        self.hypotheses = list([Hypothesis(**self.hypothesis_kwargs)])
        self.w = list([1.0])

        self.visited_blocks = set()
        self.experience = list()

    def estimate(self, list_blocks, list_arms, list_rewards):
        for b, a, r in zip(list_blocks, list_arms, list_rewards):
            self.update(b, a, r)

    def update(self, block, arm, reward):

        # special case, first trial
        if len(self.visited_blocks) == 0:
            h = self.hypotheses[0]
            h.augment_assignment(block, 0)
            # h.update(block, arm, reward)

        # here, we just calculating a forward pass of the DVPI (similar to a generalization of the local MAP
        # to a k-local MAP) and not simulating re-evaluating previously locked in particles.

        # for newly visited blocks, augment the hypothesis space with all new combinations of hypothesis
        if block not in self.visited_blocks:
            self.visited_blocks.add(block)

            augmented_hypothesis_space = list()
            augmented_hypothesis_scores = list()
            hash_keys = set()

            for h in self.hypotheses:
                max_k = max(h.cluster_assignments) + 1
                for k0 in range(max_k + 1):

                    # create a new hypothesis
                    h0 = h.deepcopy()
                    h0.augment_assignment(block, k0)

                    # update the hypothesis with the observation (so we use the posterior, not the prior)
                    h0.update(block, arm, reward)

                    if hash(h0) not in hash_keys:
                        hash_keys.add(hash(h0))
                        augmented_hypothesis_space.append(h0)
                        augmented_hypothesis_scores.append(h0.get_log_post())

            # select *up to* k MAP hypotheses
            idx = np.argsort(augmented_hypothesis_scores) < self.k

            self.hypotheses = [h for keep, h in zip(idx, augmented_hypothesis_space) if keep]

            # calculate the weights of the for the filter
            self.w = np.array([w0 for keep, w0 in zip(idx, augmented_hypothesis_scores) if keep])
            self.w -= np.max(self.w)
            self.w = np.exp(self.w - logsumexp(self.w))

        else:

            # if the block has already been observed, then we've locked the particles in
            # just update the weights.
            for h in self.hypotheses:
                h.update(block, arm, reward)

            self.w = np.array([h.get_log_post() for h in self.hypotheses])
            self.w -= np.max(self.w)
            self.w = np.exp(self.w - logsumexp(self.w))

        self.experience.append(tuple([block, arm, reward]))

    def get_nmll(self, block, arm, r):
        # I guess the thing to do is get the take the expectation over the hypothesis space:
        # print [np.log(h.get_obs_prob(block, arm, r)) for h in self.hypotheses]
        nmll_h = [np.log(w0) + h.get_obs_logprob(block, arm, r) for w0, h in zip(self.w, self.hypotheses)]
        return -logsumexp(nmll_h)

    def get_mean_stdev(self, block, arm):

        mu = 0
        var0 = 0
        var1 = 0
        var2 = 0

        # special case for new blocks that have not been seen
        if block not in self.visited_blocks:

            for w, h in zip(self.w, self.hypotheses):

                # the mean of the arm is just the weighted average of the means (with the particle weights)
                _mu0 = h.get_mean(block, arm)
                mu += w * _mu0

                # the variance calculation:
                # var(f) = sum(w*sigma^2) + sum(w*mu^2) - (sum(w*mu))^2
                var0 += w * h.get_var_prior(block, arm)
                var1 += w * (_mu0 ** 2)
                var2 += w * _mu0

        # blocks that have been seen before
        else:
            for w, h in zip(self.w, self.hypotheses):

                # the mean of the arm is just the weighted average of the means (with the particle weights)
                mu += w * h.get_mean(block, arm)

                # the variance calculation:
                # var(f) = sum(w*sigma^2) + sum(w*mu^2) - (sum(w*mu))^2
                var0 += w * h.get_var(block, arm)
                var1 += w * (h.get_mean(block, arm) ** 2)
                var2 += w * h.get_mean(block, arm)

        stdev = np.sqrt(var0 + var1 - (var2 ** 2))
        return mu, stdev

    def _get_var_prior(self, block, arm):
        """ this is a special case function that returns the prior probability of each cluster
        and it's variance for the """
        pass

    def get_model_log_prob(self):

        # get the log prob for each particle in the filter
        log_prob_h = np.array([h.get_log_post() for h in self.hypotheses])

        # normalize, exponentiate,
        prob = np.exp(log_prob_h - logsumexp(log_prob_h))

        # multiply by weights
        prob = np.multiply(prob, self.w)

        # return the total log probability
        return np.log(np.sum(prob))