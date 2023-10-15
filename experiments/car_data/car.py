import numpy as np
import utils_driving as utils
# import theano as th
# import theano.tensor as tt
import torch
from trajectory import Trajectory
import feature

class Car(object):
    def __init__(self, dyn, x0, color='yellow', T=5):
        x0 = torch.tensor(x0)
        self.data0 = {'x0': x0}
        self.bounds = [(-1., 1.), (-1., 1.)]
        self.T = T
        self.dyn = dyn
        self.traj = Trajectory(T, dyn)
        self.traj.x0[:] = x0
        self.linear = Trajectory(T, dyn)
        self.linear.x0[:] = x0
        self.color = color
        self.default_u = np.zeros(self.dyn.nu)
    def reset(self):
        self.traj.x0[:] =self.data0['x0']
        self.linear.x0[:] =self.data0['x0']
        # breakpoint()
        for t in range(self.T):
            self.traj.u[t].fill_(0.0) #np.zeros(self.dyn.nu,1))
            self.linear.u[t].fill_(0.0) #self.default_u
    def move(self):
        self.traj.tick()
        self.linear.x0[:] = torch.tensor(self.traj.x0.numpy())
    @property
    def x(self):
        return self.traj.x0.numpy()
    @x.setter
    def x(self, value):
        self.traj.x0[:] = torch.tensor(value)
    @property
    def u(self):
        return self.traj.u[0].numpy()
    @u.setter
    def u(self, value):
        self.traj.u[0][:] = torch.tensor(value)
    def control(self, steer, gas):
        pass
