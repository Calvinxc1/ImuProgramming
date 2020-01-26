import numpy as np
import torch as pt

class AdamGradient:
    def __init__(self, alpha=0.8, beta=0.9, clamp=1e-16):
        self.alpha = alpha
        self.beta = beta
        self.clamp = clamp
        
        self.grad = {}
        self.grad_sq = {}
        
    def descent(self, loss, coefs, learn_rate):
        gradient = self.grad_calc(loss, coefs)
        with pt.no_grad():
            for idx in gradient.keys():
                self.grad[idx] = (self.alpha * gradient[idx]) + ((1-self.alpha) * self.grad.get(idx, gradient[idx]))
                self.grad_sq[idx] = (self.beta * (gradient[idx]**2)) + ((1-self.beta) * self.grad_sq.get(idx, gradient[idx]**2))
                step = self.grad[idx] / pt.clamp(pt.sqrt(self.grad_sq[idx]), self.clamp, np.inf)
                coefs[idx] = coefs[idx] - (step * learn_rate)
        return coefs
    
    @staticmethod
    def grad_calc(loss, coefs):
        def coef_to_list(coefs):
            coef_idx = []
            coef_val = []
            for idx, val in coefs.items():
                coef_idx.append(idx)
                coef_val.append(val)
            return (coef_idx, coef_val)
        
        coef_idx, coef_val = coef_to_list(coefs)
        grad_val = pt.autograd.grad(loss, coef_val)
        gradient = {idx:val for idx, val in zip(coef_idx, grad_val)}
        return gradient