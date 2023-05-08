#!/usr/bin/env python3

from __future__ import annotations

import torch
from botorch.acquisition.analytic import PosteriorMean

from src.get_preferential_gp_sample import get_preferential_gp_rff_sample
from src.utils import optimize_acqf_and_get_suggested_query


def gen_thompson_sampling_query(
    model, num_alternatives, bounds, num_restarts, raw_samples
):
    query = []
    for _ in range(num_alternatives):
        model_rff_sample = get_preferential_gp_rff_sample(model=model, n_samples=1)
        acquisition_function = PosteriorMean(
            model=model_rff_sample
        )  # Approximate sample from the GP posterior
        new_x = optimize_acqf_and_get_suggested_query(
            acq_func=acquisition_function,
            bounds=bounds,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            batch_size=1,  # Batching is not supported by RFFs-based sample constructor
            batch_limit=1,
            init_batch_limit=1,
        )
        query.append(new_x.clone())

    query = torch.cat(query, dim=-2)
    query = query.unsqueeze(0)
    return query
