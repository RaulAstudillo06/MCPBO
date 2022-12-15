from copy import deepcopy
from typing import Optional

from botorch.models.model_list_gp_regression import ModelListGP
from torch import Tensor

from src.models.composite_pairwise_gp import CompositePairwiseGP
from src.models.pairwise_kernel_variational_gp import PairwiseKernelVariationalGP


def fit_model(
    queries: Tensor,
    responses: Tensor,
    model_type: str,
    likelihood: Optional[str] = "probit",
):  
    if model_type == "Standard":
        model = PairwiseKernelVariationalGP(queries, responses[..., -1])
    elif model_type == "Known_Utility":
        output_dim = responses.shape[-1] - 1
        models_list = []
        for j in range(output_dim):
            model = PairwiseKernelVariationalGP(queries, responses[..., j])
            models_list.append(deepcopy(model))
        model = ModelListGP(*models_list)
    elif model_type == "Composite":
        model = CompositePairwiseGP(queries, responses, use_attribute_uncertainty=True)
    return model
