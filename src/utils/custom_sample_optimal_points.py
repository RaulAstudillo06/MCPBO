from __future__ import annotations

from math import ceil
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from botorch.acquisition.multi_objective.utils import random_search_optimizer
from botorch.exceptions.errors import UnsupportedError
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.model import Model
from torch import Tensor

from src.utils.get_preferential_gp_sample import get_preferential_gp_rff_sample

def sample_optimal_points(
    model: Model,
    bounds: Tensor,
    num_samples: int,
    num_points: int,
    optimizer: Callable[
        [GenericDeterministicModel, Tensor, int, bool, Any], Tuple[Tensor, Tensor]
    ] = random_search_optimizer,
    num_rff_features: int = 512,
    maximize: bool = True,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Compute a collection of optimal inputs and outputs from samples of a Gaussian
    Process (GP).

    Steps:
    (1) The samples are generated using random Fourier features (RFFs).
    (2) The samples are optimized sequentially using an optimizer.

    TODO: We can generalize the GP sampling step to accommodate for other sampling
        strategies rather than restricting to RFFs e.g. decoupled sampling.

    TODO: Currently this defaults to random search optimization, might want to
        explore some other alternatives.

    Args:
        model: The model. This does not support models which include fantasy
            observations.
        bounds: A `2 x d`-dim Tensor containing the input bounds.
        num_samples: The number of GP samples.
        num_points: The number of optimal points to be outputted.
        optimizer: A callable that solves the deterministic optimization problem.
        num_rff_features: The number of random Fourier features.
        maximize: If true, we consider a maximization problem.
        optimizer_kwargs: The additional arguments for the optimizer.

    Returns:
        A two-element tuple containing

        - A `num_samples x num_points x d`-dim Tensor containing the collection of
            optimal inputs.
        - A `num_samples x num_points x M`-dim Tensor containing the collection of
            optimal objectives.
    """
    tkwargs = {"dtype": bounds.dtype, "device": bounds.device}
    M = model.num_outputs
    d = bounds.shape[-1]
    if M == 1:
        if num_points > 1:
            raise UnsupportedError(
                "For single-objective optimization `num_points` should be 1."
            )
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    pareto_sets = torch.zeros((num_samples, num_points, d), **tkwargs)
    pareto_fronts = torch.zeros((num_samples, num_points, M), **tkwargs)
    for i in range(num_samples):
        sample_i = get_preferential_gp_rff_sample(
            model=model, n_samples=1
        )
        ps_i, pf_i = optimizer(
            model=sample_i,
            bounds=bounds,
            num_points=num_points,
            maximize=maximize,
            **optimizer_kwargs,
        )
        pareto_sets[i, ...] = ps_i
        pareto_fronts[i, ...] = pf_i

    return pareto_sets, pareto_fronts