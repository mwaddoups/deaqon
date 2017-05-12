from __future__ import division
import numpy as np
from collections import deque

class RandomMemory:
    def __init__(self, mem_size):
        self.mem_size = mem_size
        self.memory = deque(maxlen=mem_size)
    
    def get_size(self):
        return len(self.memory)

    def add(self, *args):
        self.memory.append(args)

    def sample(self, size):
        idx = np.arange(len(self.memory))
        idx = np.random.choice(idx, size, replace=False)
        return map(np.vstack, zip(*np.array(self.memory)[idx]))
