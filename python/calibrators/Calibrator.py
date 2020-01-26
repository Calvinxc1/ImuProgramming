import numpy as np
import pandas as pd
from tqdm.notebook import tqdm, trange
import torch as pt

from .models import Simple
from .loss import Euclidian

class Calibrator:
    def __init__(self, mag_data, ref_data, Model=Simple, model_params={}, Loss=Euclidian, loss_params={}, verbose=False):
        self.mag_tensor = pt.from_numpy(mag_data).type(pt.Tensor)
        self.ref_scale = np.sqrt((ref_data**2).sum())
        self.verbose = verbose
        
        self.Model = Model(**model_params)
        self.Loss = Loss(**loss_params)
        
    @property
    def coefs(self):
        return self.Model.coefs
        
    def calibrate(self, iters, learn_rate):
        self.loss_rcd = []
        t = trange(iters) if self.verbose else range(iters)
        for i in t: self._iterate(t, learn_rate)
        self.mag_scaled = self.Model.calc(self.mag_tensor)
        return pd.DataFrame(self.mag_scaled.detach().cpu().numpy(), columns=['x','y','z'])
                    
    def _iterate(self, t, learn_rate):
        self.Model.init()
        
        adj_data = self.Model.calc(self.mag_tensor)
        scaled_data = pt.sqrt((adj_data**2).sum(dim=1))
        
        loss = self.Loss.calc(self.ref_scale, scaled_data)
        self.loss_rcd.append(loss.detach().cpu().numpy())
        if self.verbose: t.set_postfix({'loss': self.loss_rcd[-1]})

        self.Model.update(loss, learn_rate)