import numpy as np

class SqDecLearn:
    def __init__(self, seed=1):
        self.seed = seed
        self.i = 1
        
    @property
    def learn_rate(self):
        learn_rate = self.seed / np.sqrt(self.i)
        self.i += 1
        return learn_rate