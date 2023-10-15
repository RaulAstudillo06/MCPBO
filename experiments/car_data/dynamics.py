# import theano as th
# import theano.tensor as tt
import torch

class Dynamics(object):
    def __init__(self, nx, nu, f, dt=None):
        self.nx = nx
        self.nu = nu
        self.dt = dt
        if dt is None:
            self.f = f
        else:
            self.f = lambda x, u: x+dt*f(x, u)
    def __call__(self, x, u):
        return self.f(x, u)

class CarDynamics(Dynamics):
    # https://msl.cs.uiuc.edu/planning/node658.html#eqn:ctecar
    def __init__(self, dt=0.1, ub=[(-3., 3.), (-1., 1.)], friction=1.):
        def f(x, u):
            return torch.stack([
                x[3]*torch.cos(x[2]),
                x[3]*torch.sin(x[2]),
                x[3]*u[0],
                u[1]-x[3]*friction
            ])
        Dynamics.__init__(self, 4, 2, f, dt)

if __name__ == '__main__':
    dyn = CarDynamics(0.1)
    x = torch.tensor([0.0, 0.0, 0.0, 0.01], dtype=torch.float32) 
    u = torch.tensor([0.2,0.1], dtype=torch.float32) 
    breakpoint()
    dyn(x, u)
