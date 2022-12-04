from copy import deepcopy
from typing import Optional

from botorch.fit import fit_gpytorch_model
from botorch.models.likelihoods.pairwise import (
    PairwiseProbitLikelihood,
    PairwiseLogitLikelihood,
)
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from torch import Tensor

from src.models.composite_pairwise_gp import CompositePairwiseGP
from src.utils import training_data_for_pairwise_gp


def fit_model(
    queries: Tensor,
    responses: Tensor,
    model_type: str,
    likelihood: Optional[str] = "logit",
):
    if model_type == "Standard":
        if likelihood == "probit":
            likelihood_func = PairwiseProbitLikelihood()
        else:
            likelihood_func = PairwiseLogitLikelihood()
        datapoints, comparisons = training_data_for_pairwise_gp(
            queries, responses[:, -1]
        )
        model = PairwiseGP(
            datapoints,
            comparisons,
            likelihood=likelihood_func,
            jitter=1e-4,
        )

        mll = PairwiseLaplaceMarginalLogLikelihood(
            likelihood=model.likelihood, model=model
        )
        fit_gpytorch_model(mll)
        model = model.to(device=queries.device, dtype=queries.dtype)
    elif model_type == "Known_Utility":
        output_dim = responses.shape[-1] - 1
        models_list = []
        for j in range(output_dim):
            if likelihood == "probit":
                likelihood_func = PairwiseProbitLikelihood()
            else:
                likelihood_func = PairwiseLogitLikelihood()
            datapoints, comparisons = training_data_for_pairwise_gp(
                queries, responses[:, j]
            )
            model = PairwiseGP(
                datapoints,
                comparisons,
                likelihood=likelihood_func,
                jitter=1e-4,
            )

            mll = PairwiseLaplaceMarginalLogLikelihood(
                likelihood=model.likelihood, model=model
            )
            fit_gpytorch_model(mll)
            model = model.to(device=queries.device, dtype=queries.dtype)
            models_list.append(deepcopy(model))
        model = ModelListGP(*models_list)
    elif model_type == "Composite":
        model = CompositePairwiseGP(queries, responses, use_attribute_uncertainty=True)
    return model
