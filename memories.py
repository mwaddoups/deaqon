from __future__ import division
import numpy as np

class RandomMemory:
    def __init__(self, mem_size):
        self.mem_size = mem_size
        self.loc = 0

    def init(self, n_obs, n_act):
        self.X = np.zeros((self.mem_size, n_obs))
        self.y = np.zeros((self.mem_size, n_act))

    def add(self, x_sample, y_sample):
        loc = self.loc % self.mem_size
        self.X[loc, :] = x_sample
        self.y[loc, :] = y_sample

        self.loc += 1

    def sample(self, size):
        if self.loc < size:
            return None, None
        else:
            idx = np.arange(np.minimum(self.mem_size, self.loc + 1))
            idx = np.random.choice(idx, size, replace=False)
            return self.X[idx, :], self.y[idx, :]

class QueueMemory(RandomMemory):
    def sample(self, size):
        if self.loc < size:
            return None, None
        else:
            idx = np.arange(np.minimum(self.mem_size, self.loc + 1))
            start = (self.loc - size) % self.mem_size
            end = self.loc % self.mem_size
            idx = idx[(idx > start) & (idx < end)]
            return self.X[idx, :], self.y[idx, :]
    

class SelectiveMemory:
    def __init__(self, mem_size):
        self.mem_size = mem_size
        self.loc = 0

    def init(self, n_obs, n_act):
        self.X = np.zeros((self.mem_size, n_obs))
        self.y = np.zeros((self.mem_size, n_act))
        self.num_hits = np.ones(self.mem_size)

    def add(self, x_sample, y_sample):
        loc = self.loc % self.mem_size
        self.X[loc, :] = x_sample
        self.y[loc, :] = y_sample
        self.num_hits[loc] = 1.

        self.loc += 1

    def sample(self, size):
        idx = np.arange(np.minimum(self.mem_size, self.loc + 1))

        probs = (1. / np.array(self.num_hits[idx]))
        total_prob = np.sum(probs)
        probs = probs / total_prob
        
        idx = np.random.choice(idx, size, p=probs)
        
        self.num_hits[idx] += 1
        return self.X[idx, :], self.y[idx, :]
