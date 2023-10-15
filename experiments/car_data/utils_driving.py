import torch
import numpy as np

def extract(var):
    return var.cpu().detach().numpy()

def shape(var):
    return var.shape

def vector(n):
    return torch.zeros(n, dtype=torch.float32)

def matrix(n, m):
    return torch.zeros((n, m), dtype=torch.float32)

def grad(f, x, constants=[]):
    ret = torch.autograd.grad(f, x, torch.ones_like(f), create_graph=True, retain_graph=True)
    if isinstance(ret, tuple):
        ret = torch.cat(ret)
    return ret

def jacobian(f, x, constants=[]):
    sz = shape(f)
    return torch.stack([grad(f[i], x) for i in range(sz)])

def hessian(f, x, constants=[]):
    return jacobian(grad(f, x, constants=constants), x, constants=constants)
