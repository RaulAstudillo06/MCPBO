import numpy as np
import utils_driving as utils
import torch
from trajectory import Trajectory
import feature
import numpy as np
import scipy.optimize as opt

from world import World
import car
import dynamics

# import visualize
import lane

from test_car_sim import *

from algos import generate_psi
import algos
import numpy as np
import sys
import os, math


def getPDCtrlAction(driver_env, agent1, agent2, ctrl_param):
    # check distance with the other agent
    # if distance is less than threshold, then use avoid the agent
    # else use PD controller to go to the goal
    phi = agent1[2]
    distance_to_other_agent = torch.norm(agent1[0:2] - agent2[0:2])
    if distance_to_other_agent < ctrl_param["dist_thresh"]:
        e = agent2[0:2] - agent1[0:2]  # error in position
        K = (
            1 - np.exp(-ctrl_param["ao_scaling"] * np.linalg.norm(e) ** 2)
        ) / np.linalg.norm(
            e
        )  # Scaling for velocity
        v = np.linalg.norm(K * e)  # Velocity decreases as bot gets closer to obstacle
        phi_d = -math.atan2(e[1], e[0])  # Desired heading
        omega = ctrl_param["K_p"] * math.atan2(
            math.sin(phi_d - phi), math.cos(phi_d - phi)
        )  # Only P part of a PID controller to give omega as per desired heading

    else:
        e = ctrl_param["des_goal"] - agent1[0:2]  # error in position
        K = (
            1 - np.exp(-ctrl_param["gtg_scaling"] * np.linalg.norm(e) ** 2)
        ) / np.linalg.norm(
            e
        )  # Scaling for velocity
        v = np.linalg.norm(K * e)  # Velocity decreases as bot gets closer to goal
        phi_d = math.atan2(e[1], e[0])  # Desired heading
        omega = ctrl_param["K_p"] * math.atan2(
            math.sin(phi_d - phi), math.cos(phi_d - phi)
        )  # Only P part of a PID controller to give omega as per desired heading

    v = np.linalg.norm(K * e)  # Velocity decreases as bot gets closer to goal
    v = max(v, driver_env.robot.bounds[0][0])
    v = min(v, driver_env.robot.bounds[0][1])

    omega = omega / agent1[3]
    omega = max(omega, driver_env.robot.bounds[0][0])
    omega = min(omega, driver_env.robot.bounds[0][1])

    return [omega, v]


def getCtrlAction(driver_env, agent1, agent2, ctrl_param):
    actA, actB, _ = algos.random(driver_env)
    return actA


def evalFeatures(driver_env, ctrl_param, num_time_point=50):
    car_traj = []

    rob_x = driver_env.robot.data0["x0"]
    human_x = driver_env.human.data0["x0"]

    for i in range(num_time_point):
        # act = getCtrlAction(driver_env,rob_x,human_x,ctrl_param)
        act = getPDCtrlAction(driver_env, rob_x, human_x, ctrl_param)
        rob_x = driver_env.robot.dyn(rob_x, act)
        actRand = getCtrlAction(driver_env, rob_x, human_x, ctrl_param)
        # optimal_ctrl = compute_best(driver_env, [1,2,0.5,0.5], 5)
        human_x = driver_env.human.dyn(human_x, actRand[0:2])
        car_traj.append(np.vstack([rob_x.numpy(), human_x.numpy()]))

    # breakpoint()
    return get_features(np.array(car_traj))


def updateCtrlParam(ctrls):
    ctrl_param = {
        "des_goal": torch.tensor([0, 50]),
        "dist_thresh": ctrls[3],
        "ao_scaling": ctrls[0],
        "gtg_scaling": ctrls[1],
        "K_p": ctrls[2],
    }
    return ctrl_param


# D = 100
