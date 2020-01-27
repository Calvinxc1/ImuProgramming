import pandas as pd
import torch as pt

from .gradients import Adam

class ComplexModel:
    def __init__(self, Gradient=Adam, gradient_params={}):
        self.Gradient = Gradient(**gradient_params)
        self._coefs = {
            'offset': pt.zeros(3),
            'orth': pt.ones(3),
            'scale': pt.ones(3),
            'hard': pt.zeros(3),
            'soft': pt.eye(3),
        }
    
    @property
    def coefs(self):
        coefs = {
            key:val.detach().cpu().numpy()
            for key, val in self._coefs.items()
        }
        return coefs
    
    def init(self):
        for val in self._coefs.values(): val.requires_grad = True
        
    def calc(self, mag_tensor):
        adj_data = (mag_tensor + self._coefs['bias']) @ self._coefs['skew']
        return adj_data
    
    def update(self, loss):
        self._coefs = self.Gradient.descent(loss, self._coefs)
        