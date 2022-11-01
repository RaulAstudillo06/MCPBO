import math
import os
import sys
import torch

from botorch.settings import debug
from botorch.test_functions.synthetic import Ackley

from torch import Tensor

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)
debug._set_state(False)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
print(script_dir[:-12])
sys.path.append(script_dir[:-12])

from src.experiment_manager import experiment_manager


# Objective function
a = 20.0
b = 0.2
c = 2 * math.pi

input_dim = 6
output_dim = 2


def obj_func(X: Tensor) -> Tensor:
    X_unnorm = 4.0 * X - 2.0
    output = torch.zeros(X_unnorm.shape[:-1] + torch.Size([3]))
    for i in range(input_dim):
        output[..., 0] += X_unnorm[..., i] ** 2
        output[..., 1] += torch.cos(c * X_unnorm[..., i])
        output[..., 2] += torch.cos(c * X_unnorm[..., i])
    output /= input_dim
    return output


def utility_func(Y: Tensor) -> Tensor:
    output = (
        a * torch.exp(-b * (torch.sqrt(Y[..., 0]))) + torch.exp(Y[..., 1]) - a - math.e
    )
    return output


# Algos
algo = "qEUBO"
model_type = "Composite"

# estimate noise level
comp_noise_type = "logit"
noise_level = 0.0001

# Run experiment
if len(sys.argv) == 3:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[2])
elif len(sys.argv) == 2:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[1])

experiment_manager(
    problem="ackley",
    obj_func=obj_func,
    utility_func=utility_func,
    input_dim=input_dim,
    output_dim=3,
    comp_noise_type=comp_noise_type,
    comp_noise=noise_level,
    algo=algo,
    model_type=model_type,
    batch_size=2,
    num_init_queries=4 * input_dim,
    num_algo_iter=100,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=False,
)
