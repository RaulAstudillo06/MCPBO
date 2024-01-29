import numpy as np
import os
import sys
import torch

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.settings import debug
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
input_dim = 5
num_objectives = 2

inputs = torch.load("exo2_data/normalized_inputs.txt")
objective_vals = torch.load("exo2_data/objective_vals.txt")
model = SingleTaskGP(
    train_X=inputs,
    train_Y=objective_vals,
    outcome_transform=Standardize(num_objectives),
)
model.load_state_dict(torch.load("exo2_data/exo2_surrogate_state_dict.json"), strict=True)
model.eval()


def utility_func(X: Tensor) -> Tensor:
    return model.posterior(X).mean.detach()


# Estimate noise level
comp_noise_type = "logit"
if False:
    noise_level = get_noise_level(
        utility_func,
        input_dim,
        target_error=0.2,
        top_proportion=0.01,
        num_samples=10000,
        comp_noise_type=comp_noise_type,
    )
    print(noise_level)
    print(e)

noise_level = [0.0066, 0.0181]

# Algos
algo = "SDTS"
# algo = "I-PBO-DTS"
# algo = "Random"
# algo = "qEHVI"
# algo = "qParEGO"

# Run experiment
if len(sys.argv) == 3:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[2])
elif len(sys.argv) == 2:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[1])

experiment_manager(
    problem="exo2",
    utility_func=utility_func,
    input_dim=input_dim,
    num_attributes=num_objectives,
    obs_attributes=[False, False, False],
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
