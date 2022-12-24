from copy import deepcopy
from typing import Optional

from botorch.models.model import Model
from torch import Tensor

from src.models.composite_pairwise_gp import CompositePairwiseGP
from src.models.pairwise_kernel_variational_gp import PairwiseKernelVariationalGP


def fit_model(
    queries: Tensor,
    responses: Tensor,
    model_type: str,
    state_dict = None,
    likelihood: Optional[str] = "probit",
):  
    for i in range(10):
        try:
            if model_type == "Standard":
                model = PairwiseKernelVariationalGP(queries, responses[..., -1])
            elif model_type == "Composite":
                model = CompositePairwiseGP(queries, responses, use_attribute_uncertainty=True)
            return model
        except:
            print("Number of failed attempts to train the model: " + str(i + 1))


def get_state_dict(model: Model, model_type: str):
    if model_type == "Standard":
        state_dict = deepcopy(model.state_dict())
    elif model_type == "Composite":
        state_dict = []
        for attribute_model in model.attribute_models:
            state_dict.append(deepcopy(attribute_model.state_dict()))
        state_dict.append(deepcopy(model.utility_model[0].state_dict()))
    return state_dict


def load_state_dict(model: Model, state_dict, model_type: str):
    if model_type == "Standard":
        model.load_state_dict(state_dict)
        model.eval()
    elif model_type == "Composite":
        for i in range(len(model.attribute_models)):
            model.attribute_models[i].aux_model.load_state_dict(state_dict[i])
            model.attribute_models[i].aux_model.eval()
        model.utility_model[0].aux_model.load_state_dict(state_dict[-1])
        model.utility_model[0].aux_model.eval()
    return model
