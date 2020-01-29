import pandas as pd
import torch as pt

from .gradients import Adam

class ComplexModel:
    def __init__(self, Gradient=Adam, gradient_params={}):
        self.Gradient = Gradient(**gradient_params)
        self._coefs = {
            'bias': pt.zeros(3),
            'skew': pt.eye(3),
        }
    
    @property
    def coefs(self):
        coefs = pd.DataFrame({
            'bias': self._coefs['bias'].detach().cpu().numpy(),
            'skew_x': self._coefs['skew'][:,0].detach().cpu().numpy(),
            'skew_y': self._coefs['skew'][:,1].detach().cpu().numpy(),
            'skew_z': self._coefs['skew'][:,2].detach().cpu().numpy(),
        }, index=['x','y','z'])
        return coefs
    
    def init(self):
        for val in self._coefs.values(): val.requires_grad = True
        
    def calc(self, mag_tensor):
        adj_data = (mag_tensor @ self._coefs['skew']) + self._coefs['bias']
        return adj_data
    
    def update(self, loss):
        self._coefs = self.Gradient.descent(loss, self._coefs)
        