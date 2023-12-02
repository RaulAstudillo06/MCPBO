import os
import sys
import torch

from botorch.settings import debug
from botorch.test_functions.multi_objective import DTLZ2

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)
debug._set_state(False)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
print(script_dir[:-12])
sys.path.append(script_dir[:-12])

from src.experiment_manager import experiment_manager
from src.utils.get_noise_level import get_noise_level


# Utility function
input_dim = 3
num_attributes = 2
utility_func = DTLZ2(dim=input_dim, num_objectives=num_attributes, negate=True)

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

noise_level = [0.0033, 0.0033]

# Algos
# algo = "SDTS"
algo = "qEHVI"
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
    problem="dtlz2",
    utility_func=utility_func,
    input_dim=input_dim,
    num_attributes=num_attributes,
    obs_attributes=[False, False],
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
