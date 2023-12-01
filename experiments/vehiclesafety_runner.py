import os
import sys
import torch

from botorch.settings import debug
from botorch.test_functions.multi_objective import VehicleSafety

from torch import Tensor

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)
debug._set_state(False)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
print(script_dir[:-12])
sys.path.append(script_dir[:-12])

from src.experiment_manager import experiment_manager
from src.utils.get_noise_level import get_noise_level


# Utility function
vehiclesafety_func = VehicleSafety(negate=True)
input_dim = vehiclesafety_func.dim
num_attributes = vehiclesafety_func.num_objectives


def utility_func(X: Tensor) -> Tensor:
    X_unscaled = 2.0 * X + 1.0
    output = vehiclesafety_func(X_unscaled)
    return output


# Estimate noise level
comp_noise_type = "logit"
if False:
    noise_level = get_noise_level(
        utility_func,
        input_dim,
        target_error=0.2,
        top_proportion=0.01,
        num_samples=10000000,
        comp_noise_type=comp_noise_type,
    )
    print(noise_level)
    print(e)

noise_level = [0.6146, 0.0989, 0.0021]

# Algos
algo = "SDTS"
# algo = "I-PBO-DTS"
# algo = "Random"

# Run experiment
if len(sys.argv) == 3:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[2])
elif len(sys.argv) == 2:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[1])

experiment_manager(
    problem="vehiclesafety_mixed",
    utility_func=utility_func,
    input_dim=input_dim,
    num_attributes=num_attributes,
    obs_attributes=[False, False, True],
    comp_noise_type=comp_noise_type,
    comp_noise=noise_level,
    algo=algo,
    batch_size=2,
    num_init_queries=2 * (input_dim + 1),
    num_algo_iter=100,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=True,
)
