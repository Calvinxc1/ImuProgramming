class ConstLearn:
    def __init__(self, seed=1e-2):
        self.seed = seed
        
    @property
    def learn_rate(self):
        return self.seed