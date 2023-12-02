from typing import List, Optional

import torch

from botorch.acquisition import AcquisitionFunction, PosteriorMean, qSimpleRegret
from botorch.fit import fit_gpytorch_mll
from botorch.generation.gen import get_best_candidates
from botorch.models.model import Model
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import sample_simplex
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor
from torch.distributions import Bernoulli, Normal, Gumbel

from src.models.variational_preferential_gp import VariationalPreferentialGP
from src.models.pairwise_kernel_variational_gp import PairwiseKernelVariationalGP


def fit_model(
    queries: Tensor,
    utility_vals: Tensor,
    responses: Tensor,
    obs_attributes: List,
    model_id: int,
    algo: str,
):
    if model_id == 1:
        Model = PairwiseKernelVariationalGP
    elif model_id == 2:
        Model = VariationalPreferentialGP

    for i in range(10):
        try:
            if algo == "Random":
                model = None
            elif algo == "I-PBO-DTS":
                model = Model(queries, responses)
            elif "SDTS" in algo or algo == "qEHVI":
                models = []
                num_attributes = responses.shape[-1]
                if any(obs_attributes):
                    queries_reshaped = queries.reshape(
                        torch.Size(
                            [queries.shape[0] * queries.shape[1], queries.shape[2]]
                        )
                    )
                    utility_vals_reshaped = utility_vals.reshape(
                        torch.Size(
                            [
                                utility_vals.shape[0] * utility_vals.shape[1],
                                utility_vals.shape[2],
                            ]
                        )
                    )

                for j in range(num_attributes):
                    if obs_attributes[j]:
                        train_Yvar = torch.full_like(
                            utility_vals_reshaped[..., [j]], 1e-4
                        )
                        if True:
                            model = SingleTaskGP(
                                train_X=queries_reshaped,
                                train_Y=utility_vals_reshaped[..., [j]],
                                outcome_transform=Standardize(m=1),
                            )
                            mll = ExactMarginalLogLikelihood(model.likelihood, model)
                            fit_gpytorch_mll(mll)
                        else:
                            model = FixedNoiseGP(
                                train_X=queries_reshaped,
                                train_Y=utility_vals_reshaped[..., [j]],
                                train_Yvar=train_Yvar,
                                outcome_transform=Standardize(m=1),
                            )
                    else:
                        model = Model(queries, responses[..., j])
                    models.append(model)

                model = ModelListGP(*models)
            return model
        except Exception as error:
            print("Number of failed attempts to train the model: " + str(i + 1))
            print(error)


def generate_initial_data(
    num_queries: int,
    batch_size: int,
    input_dim: int,
    utility_func,
    comp_noise_type,
    comp_noise,
    algo,
    seed: int = None,
):
    # generates initial data
    queries = generate_random_queries(num_queries, batch_size, input_dim, seed)
    utility_vals = get_utility_vals(queries, utility_func)
    responses = generate_responses(utility_vals, comp_noise_type, comp_noise, algo)
    return queries, utility_vals, responses


def generate_random_queries(
    num_queries: int, batch_size: int, input_dim: int, seed: int = None
):
    # generates `num_queries` queries each constituted by `batch_size` points chosen uniformly at random
    if seed is not None:
        old_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        queries = torch.rand([num_queries, batch_size, input_dim])
        torch.random.set_rng_state(old_state)
    else:
        queries = torch.rand([num_queries, batch_size, input_dim])
    return queries


def get_utility_vals(queries, utility_func):
    queries_2d = queries.reshape(
        torch.Size([queries.shape[0] * queries.shape[1], queries.shape[2]])
    )

    utility_vals = utility_func(queries_2d)
    utility_vals = utility_vals.reshape(
        torch.Size([queries.shape[0], queries.shape[1], utility_vals.shape[1]])
    )
    return utility_vals


def generate_responses(utility_vals, noise_type, noise_level, algo):
    # generates simulated (noisy) comparisons based on true underlying utility values
    corrupted_utility_vals = corrupt_vals(utility_vals, noise_type, noise_level)
    if algo == "I-PBO-DTS":
        weights = sample_simplex(d=utility_vals.shape[-1]).squeeze()
        chebyshev_scalarization = get_chebyshev_scalarization(
            weights=weights,
            Y=corrupted_utility_vals[:, 0, :],
        )
        corrupted_scalarization_vals = chebyshev_scalarization(corrupted_utility_vals)
        responses = torch.argmax(corrupted_scalarization_vals, dim=-1)

    else:
        responses = torch.argmax(corrupted_utility_vals, dim=-2)
    return responses


def corrupt_vals(vals, noise_type, noise_level):
    # corrupts utility values to simulate noise in the DM's responses
    if noise_type == "noiseless":
        corrupted_vals = vals
    elif noise_type == "probit":
        normal = Normal(torch.tensor(0.0), torch.tensor(noise_level))
        noise = gumbel.sample(sample_shape=vals.shape[:-1])
        corrupted_vals = vals + noise
    elif noise_type == "logit":
        gumbel = Gumbel(torch.tensor(0.0), torch.tensor(noise_level))
        noise = gumbel.sample(sample_shape=vals.shape[:-1])
        corrupted_vals = vals + noise
    elif noise_type == "constant":
        corrupted_vals = vals.clone()
        for i in range(vals.shape[0]):
            for j in range(vals.shape[-1]):
                coin_toss = Bernoulli(noise_level).sample().item()
                if coin_toss == 1.0:
                    corrupted_vals[i, 0, j] = vals[i, 1, j]
                    corrupted_vals[i, 1, j] = vals[i, 0, j]
    return corrupted_vals


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
