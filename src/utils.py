from typing import Optional

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.generation.gen import get_best_candidates

from botorch.optim.optimize import optimize_acqf
from torch import Tensor
from torch.distributions import Bernoulli, Normal, Gumbel


def generate_initial_data(
    num_queries: int,
    batch_size: int,
    input_dim: int,
    obj_func,
    utility_func,
    comp_noise_type,
    comp_noise,
    add_baseline_point: bool,
    seed: int = None,
):
    # generate initial data

    queries = generate_random_queries(num_queries, batch_size, input_dim, seed)
    if add_baseline_point:
        queries_against_baseline = generate_queries_against_baseline(
            100, batch_size, input_dim, obj_func, seed
        )
        queries = torch.cat([queries, queries_against_baseline], dim=0)
    obj_vals, utility_vals = get_obj_and_utility_vals(queries, obj_func, utility_func)
    responses = generate_responses(obj_vals, utility_vals, comp_noise_type, comp_noise)
    return queries, obj_vals, utility_vals, responses


def generate_random_queries(
    num_queries: int, batch_size: int, input_dim: int, seed: int = None
):
    # generate `num_queries` queries each constituted by `batch_size` points chosen uniformly at random
    if seed is not None:
        old_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        queries = torch.rand([num_queries, batch_size, input_dim])
        torch.random.set_rng_state(old_state)
    else:
        queries = torch.rand([num_queries, batch_size, input_dim])
    return queries


def generate_queries_against_baseline(
    num_queries: int, batch_size: int, input_dim: int, obj_func, seed: int = None
):
    # generate `num_queries` queries each constituted by `batch_size` points chosen uniformly at random
    # random_points = generate_random_queries(10 * (2 ** input_dim), 1, input_dim, seed + 1)
    # obj_vals = get_obj_vals(random_points, obj_func).squeeze(-1)
    best_point = torch.tensor(
        [0.52] * input_dim
    )  # random_points[torch.argmax(obj_vals), ...].unsqueeze(0)
    queries = generate_random_queries(num_queries, batch_size - 1, input_dim, seed + 2)
    queries = torch.cat([best_point.expand_as(queries), queries], dim=1)
    # print(obj_func(queries))
    return queries


def get_obj_and_utility_vals(queries, obj_func, utility_func):
    queries_2d = queries.reshape(
        torch.Size([queries.shape[0] * queries.shape[1], queries.shape[2]])
    )

    obj_vals = obj_func(queries_2d)
    utility_vals = utility_func(obj_vals)
    obj_vals = obj_vals.reshape(
        torch.Size([queries.shape[0], queries.shape[1], obj_vals.shape[1]])
    )
    utility_vals = utility_vals.reshape(
        torch.Size([queries.shape[0], queries.shape[1]])
    )
    return obj_vals, utility_vals


def generate_responses(obj_vals, utility_vals, noise_type, noise_level):
    # generate simulated comparisons based on true underlying objective
    # print(obj_vals)
    corrupted_obj_vals = corrupt_vals(obj_vals, noise_type, noise_level)
    responses_obj_vals = torch.argmax(corrupted_obj_vals, dim=-2)
    # print(responses_obj_vals)
    corrupted_utility_vals = corrupt_vals(utility_vals, noise_type, noise_level)
    # print(utility_vals)
    response_utility = torch.argmax(corrupted_utility_vals, dim=-1)
    # print(response_utility)
    responses = torch.cat([responses_obj_vals, response_utility.unsqueeze(-1)], dim=-1)
    return responses


def corrupt_vals(obj_vals, noise_type, noise_level):
    if noise_type == "noiseless":
        corrupted_obj_vals = obj_vals
    elif noise_type == "probit":
        normal = Normal(torch.tensor(0.0), torch.tensor(noise_level))
        noise = normal.sample(sample_shape=obj_vals.shape)
        corrupted_obj_vals = obj_vals + noise
    elif noise_type == "logit":
        gumbel = Gumbel(torch.tensor(0.0), torch.tensor(noise_level))
        noise = gumbel.sample(sample_shape=obj_vals.shape)
        corrupted_obj_vals = obj_vals + noise
    elif noise_type == "constant":
        corrupted_obj_vals = obj_vals.clone()
        n = obj_vals.shape[0]
        for i in range(n):
            coin_toss = Bernoulli(noise_level).sample().item()
            if coin_toss == 1.0:
                corrupted_obj_vals[i, 0] = obj_vals[i, 1]
                corrupted_obj_vals[i, 1] = obj_vals[i, 0]
    return corrupted_obj_vals


def training_data_for_pairwise_gp(queries, responses):
    num_queries = queries.shape[0]
    batch_size = queries.shape[1]
    datapoints = []
    comparisons = []
    for i in range(num_queries):
        best_item_id = batch_size * i + responses[i]
        comparison = [best_item_id]
        for j in range(batch_size):
            datapoints.append(queries[i, j, :].unsqueeze(0))
            if j != responses[i]:
                comparison.append(batch_size * i + j)
        comparisons.append(torch.tensor(comparison).unsqueeze(0))

    datapoints = torch.cat(datapoints, dim=0)
    comparisons = torch.cat(comparisons, dim=0)
    return datapoints, comparisons


def optimize_acqf_and_get_suggested_query(
    acq_func: AcquisitionFunction,
    bounds: Tensor,
    batch_size: int,
    num_restarts: int,
    raw_samples: int,
    batch_initial_conditions: Optional[Tensor] = None,
    batch_limit: Optional[int] = 4,
    init_batch_limit: Optional[int] = 20,
) -> Tensor:
    """Optimizes the acquisition function, and returns the candidate solution."""

    candidates, acq_values = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        batch_initial_conditions=batch_initial_conditions,
        options={
            "batch_limit": batch_limit,
            "init_batch_limit": init_batch_limit,
            "maxiter": 100,
            "nonnegative": False,
            "method": "L-BFGS-B",
        },
        return_best_only=False,
    )

    candidates = candidates.detach()
    acq_values_sorted, indices = torch.sort(acq_values.squeeze(), descending=True)
    # print("Acquisition values:")
    # print(acq_values_sorted)
    # print("Candidates:")
    # print(candidates[indices].squeeze())
    # print(candidates.squeeze())
    new_x = get_best_candidates(batch_candidates=candidates, batch_values=acq_values)
    return new_x