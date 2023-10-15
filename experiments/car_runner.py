import os
import sys
import torch

from botorch.settings import debug
from torch import Tensor

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)
debug._set_state(False)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
print(script_dir[:-12])
sys.path.append(script_dir[:-12])
sys.path.append(script_dir[:-12] + "\experiments\car_data")

from src.experiment_manager import experiment_manager
from src.get_noise_level import get_noise_level
from car_data.genTraj import evalFeatures, updateCtrlParam
from car_data.test_car_sim import Driver


# Objective function
input_dim = 4
num_attributes = 4


def attribute_func(X: Tensor) -> Tensor:
    X_unscaled = X.clone()
    X_unscaled[..., :3] = 100 * X_unscaled[..., :3]
    X_unscaled[..., 3] = 10 * X_unscaled[..., 3]
    fX = []
    for i in range(X_unscaled.shape[0]):
        features = []
        driver_env = Driver()
        for _ in range(100):
            ctrl_param = updateCtrlParam(X_unscaled[i, ...])
            features.append(
                torch.tensor(evalFeatures(driver_env, ctrl_param)).unsqueeze(0)
            )
        features = torch.cat(features, dim=0)
        features = features.mean(dim=0, keepdim=True)
        fX.append(features)
    fX = torch.cat(fX, dim=0)
    return fX


# Algos
algo = "SDTS"
# algo = "SDTS-HS"
# algo = "I-PBO-DTS"
# algo = "Random"

# Estimate noise level
comp_noise_type = "logit"

if False:
    noise_level = get_noise_level(
        attribute_func,
        input_dim,
        target_error=0.2,
        top_proportion=0.01,
        num_samples=10000,
        comp_noise_type=comp_noise_type,
    )
    print(noise_level)
    print(e)

noise_level = [0.001, 0.001, 0.001, 0.001]

# Run experiment
if len(sys.argv) == 3:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[2])
elif len(sys.argv) == 2:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[1])

experiment_manager(
    problem="car",
    attribute_func=attribute_func,
    input_dim=input_dim,
    num_attributes=num_attributes,
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
