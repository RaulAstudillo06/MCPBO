#!/usr/bin/env python3

from typing import Callable, Dict, Optional

import numpy as np
import os
import sys
import time
import torch
from botorch.acquisition import PosteriorMean, qSimpleRegret
from botorch.acquisition.objective import GenericMCObjective
from botorch.models.model import Model
from botorch.sampling.samplers import SobolQMCNormalSampler
from torch import Tensor

from src.acquisition_functions.eubo import qExpectedUtilityBestOption
from src.acquisition_functions.thompson_sampling import gen_thompson_sampling_query
from src.fit_model import fit_model
from src.utils import (
    generate_initial_data,
    generate_random_queries,
    get_attribute_and_utility_vals,
    generate_responses,
    optimize_acqf_and_get_suggested_query,
)

# this function runs a single trial of a given problem
# see more details about the arguments in experiment_manager.py 
def mcpbo_trial(
    problem: str,
    attribute_func: Callable,
    utility_func: Callable,
    input_dim: int,
    num_attributes: int,
    comp_noise_type: str,
    comp_noise: float,
    algo: str,
    batch_size: int,
    num_init_queries: int,
    num_algo_iter: int,
    trial: int,
    restart: bool,
    model_type: str,
    add_baseline_point: bool,
    ignore_failures: bool,
    algo_params: Optional[Dict] = None,
) -> None:

    algo_id = algo + "_" + model_type

    # Get script directory
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    project_path = script_dir[:-11]
    results_folder = (
        project_path + "/experiments/results/" + problem + "/" + algo_id + "/"
    )

    if restart:
        # check if training data is available
        try:
            # current available iterations
            queries = np.loadtxt(
                results_folder + "queries/queries_" + str(trial) + ".txt"
            )
            queries = queries.reshape(
                queries.shape[0], batch_size, int(queries.shape[1] / batch_size)
            )
            queries = torch.tensor(queries)
            attribute_vals = torch.tensor(
                np.loadtxt(results_folder + "attribute_vals/attribute_vals_" + str(trial) + ".txt")
            )
            attribute_vals = attribute_vals.reshape(
                attribute_vals.shape[0], batch_size, int(attribute_vals.shape[1] / batch_size)
            )
            utility_vals = torch.tensor(
                np.loadtxt(
                    results_folder + "utility_vals/utility_vals_" + str(trial) + ".txt"
                )
            )
            responses = torch.tensor(
                np.loadtxt(
                    results_folder + "responses/responses_" + str(trial) + ".txt"
                )
            )
            # historical max utility values within queries
            max_utility_vals_within_queries = list(
                np.loadtxt(
                    results_folder
                    + "max_utility_vals_within_queries_"
                    + str(trial)
                    + ".txt"
                )
            )
            # historical utility values at the maximum of the posterior mean
            utility_vals_at_max_post_mean = list(
                np.loadtxt(
                    results_folder
                    + "utility_vals_at_max_post_mean_"
                    + str(trial)
                    + ".txt"
                )
            )
            # historical acquisition runtimes
            runtimes = list(
                np.atleast_1d(
                    np.loadtxt(
                        results_folder + "runtimes/runtimes_" + str(trial) + ".txt"
                    )
                )
            )

            # fit model
            t0 = time.time()
            model = fit_model(
                queries,
                responses,
                model_type=model_type,
                likelihood=comp_noise_type,
            )
            t1 = time.time()
            model_training_time = t1 - t0

            iteration = len(max_utility_vals_within_queries) - 1
            print("Restarting experiment from available data.")

        except:
            # initial data
            queries, attribute_vals, utility_vals, responses = generate_initial_data(
                num_queries=num_init_queries,
                batch_size=batch_size,
                input_dim=input_dim,
                attribute_func=attribute_func,
                utility_func=utility_func,
                comp_noise_type=comp_noise_type,
                comp_noise=comp_noise,
                add_baseline_point=add_baseline_point,
                seed=trial,
            )

            # fit model
            t0 = time.time()
            model = fit_model(
                queries,
                responses,
                model_type=model_type,
                likelihood=comp_noise_type,
            )
            t1 = time.time()
            model_training_time = t1 - t0

            # historical utility values at the maximum of the posterior mean
            utility_val_at_max_post_mean = compute_utility_val_at_max_post_mean(
                attribute_func=attribute_func,
                utility_func=utility_func,
                model=model,
                model_type=model_type,
                input_dim=input_dim,
            )
            utility_vals_at_max_post_mean = [utility_val_at_max_post_mean]

            # historical max utility values within queries and runtimes
            max_utility_val_within_queries = utility_vals.max().item()
            max_utility_vals_within_queries = [max_utility_val_within_queries]

            # Historical acquisition runtimes
            runtimes = []

            iteration = 0
    else:
        # initial data
        queries, attribute_vals, utility_vals, responses = generate_initial_data(
            num_queries=num_init_queries,
            batch_size=batch_size,
            input_dim=input_dim,
            attribute_func=attribute_func,
            utility_func=utility_func,
            comp_noise_type=comp_noise_type,
            comp_noise=comp_noise,
            add_baseline_point=add_baseline_point,
            seed=trial,
        )

        # fit model
        t0 = time.time()
        model = fit_model(
            queries,
            responses,
            model_type=model_type,
            likelihood=comp_noise_type,
        )
        t1 = time.time()
        model_training_time = t1 - t0

        # historical utility values at the maximum of the posterior mean
        utility_val_at_max_post_mean = compute_utility_val_at_max_post_mean(
            attribute_func=attribute_func,
            utility_func=utility_func,
            model=model,
            model_type=model_type,
            input_dim=input_dim,
        )
        utility_vals_at_max_post_mean = [utility_val_at_max_post_mean]

        # historical max utility values within queries and runtimes
        max_utility_val_within_queries = utility_vals.max().item()
        max_utility_vals_within_queries = [max_utility_val_within_queries]

        # historical acquisition runtimes
        runtimes = []

        iteration = 0

    while iteration < num_algo_iter:
        iteration += 1
        print("Problem: " + problem)
        print("Sampling policy: " + algo_id)
        print("Trial: " + str(trial))
        print("Iteration: " + str(iteration))

        # new suggested query
        t0 = time.time()
        new_query = get_new_suggested_query(
            algo=algo,
            model=model,
            utility_func=utility_func,
            batch_size=batch_size,
            input_dim=input_dim,
            algo_params=algo_params,
            model_type=model_type,
        )
        t1 = time.time()
        acquisition_time = t1 - t0
        runtimes.append(acquisition_time + model_training_time)

        # get response at new query
        (
            new_attribute_vals,
            new_utility_val,
        ) = get_attribute_and_utility_vals(new_query, attribute_func, utility_func)
        new_responses = generate_responses(
            new_attribute_vals,
            new_utility_val,
            noise_type=comp_noise_type,
            noise_level=comp_noise,
        )

        # update training data ()
        queries = torch.cat((queries, new_query))
        attribute_vals = torch.cat([attribute_vals, new_attribute_vals], 0)
        utility_vals = torch.cat([utility_vals, new_utility_val], 0)
        responses = torch.cat((responses, new_responses))

        # fit model
        t0 = time.time()
        model = fit_model(
            queries,
            responses,
            model_type=model_type,
            likelihood=comp_noise_type,
        )
        t1 = time.time()
        model_training_time = t1 - t0

        # compute and append current utility value at the maximum of the posterior mean
        utility_val_at_max_post_mean = compute_utility_val_at_max_post_mean(
            attribute_func=attribute_func,
            utility_func=utility_func,
            model=model,
            model_type=model_type,
            input_dim=input_dim,
        )
        utility_vals_at_max_post_mean.append(utility_val_at_max_post_mean)
        print(
            "Utility value at the maximum of the posterior mean: "
            + str(utility_val_at_max_post_mean)
        )

        # append current max utility val within queries
        max_utility_val_within_queries = utility_vals.max().item()
        max_utility_vals_within_queries.append(max_utility_val_within_queries)
        print(
            "Max utility value within queries: " + str(max_utility_val_within_queries)
        )

        # save data
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        if not os.path.exists(results_folder + "queries/"):
            os.makedirs(results_folder + "queries/")
        if not os.path.exists(results_folder + "attribute_vals/"):
            os.makedirs(results_folder + "attribute_vals/")
        if not os.path.exists(results_folder + "utility_vals/"):
            os.makedirs(results_folder + "utility_vals/")
        if not os.path.exists(results_folder + "responses/"):
            os.makedirs(results_folder + "responses/")
        if not os.path.exists(results_folder + "runtimes/"):
            os.makedirs(results_folder + "runtimes/")

        queries_reshaped = queries.numpy().reshape(queries.shape[0], -1)
        np.savetxt(
            results_folder + "queries/queries_" + str(trial) + ".txt", queries_reshaped
        )
        attribute_vals_reshaped = attribute_vals.numpy().reshape(attribute_vals.shape[0], -1)
        np.savetxt(results_folder + "attribute_vals/attribute_vals_" + str(trial) + ".txt", attribute_vals_reshaped)
        np.savetxt(
            results_folder + "utility_vals/utility_vals_" + str(trial) + ".txt",
            utility_vals.numpy(),
        )
        np.savetxt(
            results_folder + "responses/responses_" + str(trial) + ".txt",
            responses.numpy(),
        )
        np.savetxt(
            results_folder + "runtimes/runtimes_" + str(trial) + ".txt",
            np.atleast_1d(runtimes),
        )
        np.savetxt(
            results_folder + "utility_vals_at_max_post_mean_" + str(trial) + ".txt",
            np.atleast_1d(utility_vals_at_max_post_mean),
        )
        np.savetxt(
            results_folder + "max_utility_vals_within_queries_" + str(trial) + ".txt",
            np.atleast_1d(max_utility_vals_within_queries),
        )

# computes the new query to be shown to the DM
def get_new_suggested_query(
    algo: str,
    model: Model,
    utility_func: Callable,
    batch_size,
    input_dim: int,
    model_type: str,
    algo_params: Optional[Dict] = None,
) -> Tensor:

    standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim])
    num_restarts = 4 * input_dim
    raw_samples = 120 * input_dim
    batch_initial_conditions = None

    if algo == "Random":
        return generate_random_queries(
            num_queries=1, batch_size=batch_size, input_dim=input_dim
        )
    elif algo == "qEUBO":
        sampler = SobolQMCNormalSampler(num_samples=64, collapse_batch_dims=True)
        if model_type == "Standard" or model_type == "Composite":
            acquisition_function = qExpectedUtilityBestOption(
                model=model, sampler=sampler
            )
        elif model_type == "Known_Utility":
            acqf_obejctive = GenericMCObjective(objective=utility_func)
            acquisition_function = qExpectedUtilityBestOption(
                model=model,
                objective=acqf_obejctive,
                sampler=sampler,
            )

    elif algo == "qTS":
        standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim])
        return gen_thompson_sampling_query(
            model, batch_size, standard_bounds, num_restarts, raw_samples
        )

    new_query = optimize_acqf_and_get_suggested_query(
        acq_func=acquisition_function,
        bounds=standard_bounds,
        batch_size=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        batch_initial_conditions=batch_initial_conditions,
    )

    new_query = new_query.unsqueeze(0)
    return new_query

# computes the (true underlying) utility value at the maximum of the posterior mean
def compute_utility_val_at_max_post_mean(
    attribute_func: Callable,
    utility_func: Callable,
    model: Model,
    model_type,
    input_dim: int,
) -> Tensor:

    standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim])
    num_restarts = 4 * input_dim
    raw_samples = 120 * input_dim

    if model_type == "Standard":
        post_mean_func = PosteriorMean(model=model)
    elif model_type == "Known_Utility":
        sampler = SobolQMCNormalSampler(num_samples=64, collapse_batch_dims=True)
        acqf_objective = GenericMCObjective(objective=utility_func)
        post_mean_func = qSimpleRegret(
            model=model, objective=acqf_objective, sampler=sampler
        )
    elif model_type == model_type == "Composite":
        sampler = SobolQMCNormalSampler(num_samples=64, collapse_batch_dims=True)
        post_mean_func = qSimpleRegret(model=model, sampler=sampler)
    max_post_mean_func = optimize_acqf_and_get_suggested_query(
        acq_func=post_mean_func,
        bounds=standard_bounds,
        batch_size=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )

    utility_val_at_max_post_mean_func = utility_func(
        attribute_func(max_post_mean_func)
    ).item()
    return utility_val_at_max_post_mean_func
