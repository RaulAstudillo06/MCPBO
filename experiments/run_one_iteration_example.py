import math
import numpy as np
import os
import sys
import torch
torch.set_default_dtype(torch.float64)

from torch import Tensor

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(script_dir[:-12])

from src.utils import (
    generate_random_queries,
    get_attribute_and_utility_vals,
    generate_responses
)
from src.run_one_iteration import run_one_iteration



# attribute and utility functions
a = 20.0
b = 0.2
c = 2 * math.pi

input_dim = 6
num_attributes = 2


def attribute_func(X: Tensor) -> Tensor:
    X_unnorm = 4.0 * X - 2.0
    output = torch.zeros(X_unnorm.shape[:-1] + torch.Size([num_attributes]))
    for i in range(input_dim):
        output[..., 0] += X_unnorm[..., i] ** 2
        output[..., 1] += torch.cos(c * X_unnorm[..., i])
    output /= input_dim
    return output


def utility_func(Y: Tensor) -> Tensor:
    output = (
        a * torch.exp(-b * (torch.sqrt(Y[..., 0]))) + torch.exp(Y[..., 1]) - a - math.e
    )
    return output

# trial id
trial = 1

# generate initial data
queries = generate_random_queries(4 * input_dim, batch_size=2, input_dim=input_dim, seed=trial)
attribute_vals, utility_vals = get_attribute_and_utility_vals(queries, attribute_func, utility_func)
responses = generate_responses(attribute_vals, utility_vals, "probit", 1e-3)

# algo
algo = "qEUBO"
model_type = "Composite"

# save initial data
problem = "example"
algo_id = algo + "_" + model_type
results_folder = (
        script_dir + "/results/" + problem + "/" + algo_id + "/"
    )
if not os.path.exists(results_folder):
    os.makedirs(results_folder + "queries/")
    os.makedirs(results_folder + "attribute_vals/")
    os.makedirs(results_folder + "utility_vals/")
    os.makedirs(results_folder + "responses/")
queries_reshaped = queries.numpy().reshape(queries.shape[0], -1)
np.savetxt(results_folder + "queries/queries_" + str(trial) + ".txt", queries_reshaped)
attribute_vals_reshaped = attribute_vals.numpy().reshape(attribute_vals.shape[0], -1)
np.savetxt(results_folder + "attribute_vals/attribute_vals_" + str(trial) + ".txt", attribute_vals_reshaped)
np.savetxt(results_folder + "utility_vals/utility_vals_" + str(trial) + ".txt", utility_vals.numpy())
np.savetxt(results_folder + "responses/responses_" + str(trial) + ".txt", responses.numpy())

# run mcpbo loop
num_iter = 2

for i in range(num_iter):
    new_query = run_one_iteration(
            queries=queries,
            responses=responses, 
            input_dim=input_dim, 
            algo=algo, 
            model_type=model_type,
            trial=trial,
            iteration=i + 1,
            results_folder=results_folder
    )
    new_attribute_vals, new_utility_val = get_attribute_and_utility_vals(new_query, attribute_func, utility_func)
    new_responses = generate_responses(new_attribute_vals, new_utility_val, noise_type="probit", noise_level=1e-3)
    # update training data
    queries = torch.cat((queries, new_query))
    attribute_vals = torch.cat([attribute_vals, new_attribute_vals], 0)
    utility_vals = torch.cat([utility_vals, new_utility_val], 0)
    responses = torch.cat((responses, new_responses))
    # save data
    queries_reshaped = queries.numpy().reshape(queries.shape[0], -1)
    np.savetxt(results_folder + "queries/queries_" + str(trial) + ".txt", queries_reshaped)
    attribute_vals_reshaped = attribute_vals.numpy().reshape(attribute_vals.shape[0], -1)
    np.savetxt(results_folder + "attribute_vals/attribute_vals_" + str(trial) + ".txt", attribute_vals_reshaped)
    np.savetxt(results_folder + "utility_vals/utility_vals_" + str(trial) + ".txt", utility_vals.numpy())
    np.savetxt(results_folder + "responses/responses_" + str(trial) + ".txt", responses.numpy())
