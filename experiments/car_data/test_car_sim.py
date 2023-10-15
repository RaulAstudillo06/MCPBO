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
import visualize
import lane

class Simulation(object):
    def __init__(self, name, total_time=1000, recording_time=[0,1000]):
        self.name = name.lower()
        self.total_time = total_time
        self.recording_time = [max(0,recording_time[0]), min(total_time,recording_time[1])]
        self.frame_delay_ms = 0

    def reset(self):
        self.trajectory = []
        self.alreadyRun = False
        self.ctrl_array = [[0]*self.input_size]*self.total_time

    @property
    def ctrl(self):
        return self.ctrl_array 
    @ctrl.setter
    def ctrl(self, value):
        self.reset()
        self.ctrl_array = value.copy()
        self.run(reset=False)

class DrivingSimulation(Simulation):
    def __init__(self, name, total_time=50, recording_time=[0,50]):
        super(DrivingSimulation, self).__init__(name, total_time=total_time, recording_time=recording_time)
        self.world = World()
        clane = lane.StraightLane([0., -1.], [0., 1.], 0.17)
        self.world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
        self.world.roads += [clane]
        self.world.fences += [clane.shifted(2), clane.shifted(-2)]
        self.dyn = dynamics.CarDynamics(0.1)
        self.robot = car.Car(self.dyn, [0., -0.3, np.pi/2., 0.4], color='orange')
        self.human = car.Car(self.dyn, [0.17, 0., np.pi/2., 0.41], color='white')
        self.world.cars.append(self.robot)
        self.world.cars.append(self.human)
        self.initial_state = [self.robot.x, self.human.x]
        self.input_size = 2
        self.reset()
        self.viewer = None

    def initialize_positions(self):
        self.robot_history_x = []
        self.robot_history_u = []
        self.human_history_x = []
        self.human_history_u = []
        self.robot.x = self.initial_state[0]
        self.human.x = self.initial_state[1]

    def reset(self):
        super(DrivingSimulation, self).reset()
        self.initialize_positions()

    def run(self, reset=False):
        
        if reset:
            self.reset()
        else:
            self.initialize_positions()
        for i in range(self.total_time):
            # breakpoint()
            self.robot.u = self.ctrl_array[i]
            if i < self.total_time//5:
                self.human.u = [0, self.initial_state[1][3]]
            elif i < 2*self.total_time//5:
                self.human.u = [1., self.initial_state[1][3]]
            elif i < 3*self.total_time//5:
                self.human.u = [-1., self.initial_state[1][3]]
            elif i < 4*self.total_time//5:
                self.human.u = [0, self.initial_state[1][3]*1.3]
            else:
                self.human.u = [0, self.initial_state[1][3]*1.3]
            self.robot_history_x.append(self.robot.x)
            self.robot_history_u.append(self.robot.u)
            self.human_history_x.append(self.human.x)
            self.human_history_u.append(self.human.u)
            self.robot.move()
            self.human.move()
            self.trajectory.append([self.robot.x, self.human.x])
        self.alreadyRun = True

    # I keep all_info variable for the compatibility with mujoco wrapper
    def get_trajectory(self, all_info=True):
        if not self.alreadyRun:
            self.run()
        return self.trajectory.copy()

    def get_recording(self, all_info=True):
        traj = self.get_trajectory(all_info=all_info)
        return traj[self.recording_time[0]:self.recording_time[1]]

    def watch(self, repeat_count=1):
        self.robot.x = self.initial_state[0]
        self.human.x = self.initial_state[1]
        breakpoint()
        if self.viewer is None:
            self.viewer = visualize.Visualizer(0.1, magnify=1.2)
            self.viewer.main_car = self.robot
            self.viewer.use_world(self.world)
            self.viewer.paused = True
        for _ in range(repeat_count):
            self.viewer.run_modified(history_x=[self.robot_history_x, self.human_history_x], history_u=[self.robot_history_u, self.human_history_u])
        self.viewer.window.close()
        self.viewer = None



class Driver(DrivingSimulation):
    def __init__(self, total_time=50, recording_time=[0,50]):
        super(Driver ,self).__init__(name='driver', total_time=total_time, recording_time=recording_time)
        self.ctrl_size = 10
        self.state_size = 0
        self.feed_size = self.ctrl_size + self.state_size
        self.ctrl_bounds = [(-1,1)]*self.ctrl_size
        self.state_bounds = []
        self.feed_bounds = self.state_bounds + self.ctrl_bounds
        self.num_of_features = 4

    def get_features(self):
        recording = self.get_recording(all_info=False)
        recording = np.array(recording)

        # staying in lane (higher is better)
        staying_in_lane = np.mean(np.exp(-30*np.min([np.square(recording[:,0,0]-0.17), np.square(recording[:,0,0]), np.square(recording[:,0,0]+0.17)], axis=0))) / 0.15343634

        # keeping speed (lower is better)
        keeping_speed = np.mean(np.square(recording[:,0,3]-1)) / 0.42202643

        # heading (higher is better)
        heading = np.mean(np.sin(recording[:,0,2])) / 0.06112367

        # collision avoidance (lower is better)
        collision_avoidance = np.mean(np.exp(-(7*np.square(recording[:,0,0]-recording[:,1,0])+3*np.square(recording[:,0,1]-recording[:,1,1])))) / 0.15258019

        return [staying_in_lane, keeping_speed, heading, collision_avoidance]

    @property
    def state(self):
        return [self.robot.x, self.human.x]
    @state.setter
    def state(self, value):
        self.reset()
        self.initial_state = value.copy()

    def set_ctrl(self, value):
        arr = [[0]*self.input_size]*self.total_time
        interval_count = len(value)//self.input_size
        interval_time = int(self.total_time / interval_count)
        arr = np.array(arr).astype(float)
        j = 0
        for i in range(interval_count):
            arr[i*interval_time:(i+1)*interval_time] = [value[j], value[j+1]]
            j += 2
        self.ctrl = list(arr)

    def feed(self, value):
        ctrl_value = value[:]
        self.set_ctrl(ctrl_value)

def func(ctrl_array, *args):
	# breakpoint()
	simulation_object = args[0]
	w = np.array(args[1])
	simulation_object.set_ctrl(ctrl_array)
	features = simulation_object.get_features()
	return -np.mean(np.array(features).dot(w))


def compute_best(simulation_object, w, iter_count=10):
	u = simulation_object.ctrl_size
	lower_ctrl_bound = [x[0] for x in simulation_object.ctrl_bounds]
	upper_ctrl_bound = [x[1] for x in simulation_object.ctrl_bounds]
	opt_val = np.inf
	for _ in range(iter_count):
		temp_res = opt.fmin_l_bfgs_b(func, x0=np.random.uniform(low=lower_ctrl_bound, high=upper_ctrl_bound, size=(u)), args=(simulation_object, w), bounds=simulation_object.ctrl_bounds, approx_grad=True)
		if temp_res[1] < opt_val:
			optimal_ctrl = temp_res[0]
			opt_val = temp_res[1]
	print(-opt_val)
	return optimal_ctrl

def play(simulation_object, optimal_ctrl):
	simulation_object.set_ctrl(optimal_ctrl)
	keep_playing = 'y'
	while keep_playing == 'y':
		keep_playing = 'u'
		simulation_object.watch(1)
		while keep_playing != 'n' and keep_playing != 'y':
			keep_playing = input('Again? [y/n]: ').lower()
	return optimal_ctrl



def get_features(traj):
        recording = traj

        # staying in lane (higher is better)
        staying_in_lane = np.mean(np.exp(-30*np.min([np.square(recording[:,0,0]-0.17), np.square(recording[:,0,0]), np.square(recording[:,0,0]+0.17)], axis=0))) / 0.15343634

        # keeping speed (lower is better)
        keeping_speed = np.mean(np.square(recording[:,0,3]-1)) / 0.42202643

        # heading (higher is better)
        heading = np.mean(np.sin(recording[:,0,2])) / 0.06112367

        # collision avoidance (lower is better)
        collision_avoidance = np.mean(np.exp(-(7*np.square(recording[:,0,0]-recording[:,1,0])+3*np.square(recording[:,0,1]-recording[:,1,1])))) / 0.15258019

        return [staying_in_lane, keeping_speed, heading, collision_avoidance]

# driver_env = Driver()
# driver_env.run()
# # breakpoint()
# features = driver_env.get_features()
# for i in range(5):
#     optimal_ctrl = compute_best(driver_env, [1,1,1,1], 20)
#     play(driver_env, optimal_ctrl)