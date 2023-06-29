import os
import sys
import torch

from botorch.settings import debug
from botorch.test_functions.multi_objective import VehicleSafety

from torch import Tensor

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)
debug._set_state(False)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
print(script_dir[:-12])
sys.path.append(script_dir[:-12])

from src.experiment_manager import experiment_manager

# from kumaraswamy_utility import KumaraswamyCDFProduct
from piecewiselinear_utility import PiecewiseLinear


# Objective function
vehiclesafety_func = VehicleSafety(negate=True)
input_dim = vehiclesafety_func.dim
num_attributes = vehiclesafety_func.num_objectives


def attribute_func(X: Tensor) -> Tensor:
    X_unscaled = 2.0 * X + 1.0
    output = vehiclesafety_func(X_unscaled)
    return output


# normalized_attribute_bounds = torch.tensor(
# [
# [0, 0, 0],
# [1, 1, 1],
# ]
# )
# concentration1 = torch.tensor([0.5, 1, 1.5])
# concentration2 = torch.tensor([1.0, 2.0, 3.0])

# utility_func = KumaraswamyCDFProduct(
# concentration1=concentration1, concentration2=concentration2, Y_bounds=normalized_attribute_bounds
# )

beta1 = torch.tensor([2, 6, 8])
beta2 = torch.tensor([1, 2, 2])
thresholds = torch.tensor([0.5, 0.8, 0.8])
utility_func = PiecewiseLinear(beta1=beta1, beta2=beta2, thresholds=thresholds)

# Algos
algo = "ScalarizedTS"
model_type = "Multioutput"

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
    problem="vehiclesafety",
    attribute_func=attribute_func,
    utility_func=utility_func,
    input_dim=input_dim,
    num_attributes=num_attributes,
    comp_noise_type=comp_noise_type,
    comp_noise=noise_level,
    algo=algo,
    model_type=model_type,
    batch_size=2,
    num_init_queries=2 * (input_dim + 1),
    num_algo_iter=100,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=True,
)
