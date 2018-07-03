import numpy as np


class MeanTracker(object):
    """ simple maximum likelihood mean tracker"""

    def __init__(self, mu_prior=25, n_arms=8):

        self.mu_prior = mu_prior
        self.n_arms = n_arms

        self.means = mu_prior * np.ones((0, n_arms))
        self.counts = np.zeros((0, n_arms), dtype=int)
        self.total_reward = np.zeros((0, n_arms), dtype=float)

        self.visited_blocks = dict()

    def update(self, block, arm, reward):
        if block not in self.visited_blocks.keys():
            self.visited_blocks[block] = len(self.visited_blocks)

            self.means = np.concatenate([self.counts, self.mu_prior * np.ones((1, self.n_arms), dtype=int)])
            self.counts = np.concatenate([self.counts, np.zeros((1, self.n_arms), dtype=int)])
            self.total_reward = np.concatenate([self.total_reward, np.zeros((1, self.n_arms), dtype=int)])

        k = self.visited_blocks[block]

        self.counts[k, arm] += 1
        self.total_reward[k, arm] += reward
        self.means[k, arm] = self.total_reward[k, arm] / self.counts[k, arm]

    def get_mean(self, block, arm):
        if block not in self.visited_blocks.keys():
            return self.mu_prior
        return self.means[self.visited_blocks[block], arm]
