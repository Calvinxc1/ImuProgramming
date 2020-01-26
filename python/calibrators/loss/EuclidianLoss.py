import torch as pt

class EuclidianLoss:
    def calc(self, actual, observed):
        error = actual - observed
        loss = pt.sqrt((error**2).mean())
        return loss