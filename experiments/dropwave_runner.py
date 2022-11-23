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
input_dim = 2
output_dim = 1


def obj_func(X: Tensor) -> Tensor:
    X_unscaled = 10.24 * X - 5.12
    input_shape = X_unscaled.shape
    output = torch.empty(input_shape[:-1] + torch.Size([output_dim]))
    norm_X = torch.norm(X_unscaled, dim=-1)
    output[..., 0] = norm_X
    return output


def utility_func(Y: Tensor) -> Tensor:
    output = (1.0 + torch.cos(12.0 * Y)) /(2.0 + 0.5 * (Y ** 2))
    output = output.squeeze(dim=-1)
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
    problem="dropwave",
    obj_func=obj_func,
    utility_func=utility_func,
    input_dim=input_dim,
    output_dim=output_dim,
    comp_noise_type=comp_noise_type,
    comp_noise=noise_level,
    algo=algo,
    model_type=model_type,
    batch_size=2,
    num_init_queries=4 * input_dim,
    num_algo_iter=10,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=False,
)
