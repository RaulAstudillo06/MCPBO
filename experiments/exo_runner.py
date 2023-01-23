import numpy as np
import os
import sys
import torch

from botorch.settings import debug
from copy import deepcopy
from torch import Tensor

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)
debug._set_state(False)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
print(script_dir[:-12])
sys.path.append(script_dir[:-12])

from src.experiment_manager import experiment_manager
from src.models.pairwise_kernel_variational_gp import PairwiseKernelVariationalGP


# Objective function
input_dim = 5
num_attributes = 3

attribute_surrogates = []

queries = np.loadtxt("queries.txt")
queries = queries.reshape(queries.shape[0], 2, int(queries.shape[1] / 2))
queries = torch.tensor(queries)
responses = torch.tensor(np.loadtxt("responses.txt"))

for i in range(3):
    model = PairwiseKernelVariationalGP(queries, responses[..., i], fit_aux_model_flag=False)
    model.load_state_dict(torch.load("exo_surrogate_state_dict_" + str(i) + ".json"), strict=True)
    model.eval()
    attribute_surrogates.append(deepcopy(model))

def attribute_func(X: Tensor) -> Tensor:
    output = []
    for i in range(3):
        output.append(attribute_surrogates[i](X).mean.detach())
    output = torch.cat(output, dim=-1)
    return output

target_vector = torch.tensor([-0.5826,  0.1094, -0.1175])

def utility_func(Y: Tensor) -> Tensor:
    output = -((Y- target_vector)**2).sum(dim=-1)
    return output

# Algos
algo = "qEUBO"
model_type = "Standard"

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
    problem="exo",
    attribute_func=attribute_func,
    utility_func=utility_func,
    input_dim=input_dim,
    num_attributes=num_attributes,
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