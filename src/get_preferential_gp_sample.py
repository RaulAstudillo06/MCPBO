import torch

from botorch.utils.gp_sampling import get_gp_samples
from copy import copy

from src.models.variational_preferential_gp import VariationalPreferentialGP
from src.models.pairwise_kernel_variational_gp import PairwiseKernelVariationalGP


def get_preferential_gp_rff_sample(model, n_samples):
    model = model.eval()
    # Adapt the model so that it has training inputs and outputs. This is required to draw RFFs-based samples
    if isinstance(model, PairwiseKernelVariationalGP):
        adapted_model = copy(model)
        queries = adapted_model.queries.clone()
        queries_items = queries.view(
            (queries.shape[0] * queries.shape[1], queries.shape[2])
        )
        adapted_model.train_inputs = [queries_items]
        sample_at_queries_items = adapted_model.posterior(queries_items).sample()
        adapted_model.train_targets = sample_at_queries_items.view(
            (queries_items.shape[0],)
        )
        adapted_model.covar_module = adapted_model.aux_model.covar_module.latent_kernel

        # This is used to draw RFFs-based samples. We set it close to zero because we want noise-free samples
        class LikelihoodForRFF:
            noise = torch.tensor(1e-4).double()

        adapted_model.likelihood = LikelihoodForRFF()
    elif isinstance(model, VariationalPreferentialGP):
        adapted_model = copy(model)
        queries_items = adapted_model.train_inputs[0]
        sample_at_queries_items = adapted_model.posterior(queries_items).sample()
        sample_at_queries_items = sample_at_queries_items.view(
            (queries_items.shape[0],)
        )
        adapted_model.train_targets = sample_at_queries_items
    # Draw RFFs-based (approximate) GP sample
    gp_samples = get_gp_samples(
        model=adapted_model,
        num_outputs=1,
        n_samples=n_samples,
        num_rff_features=1000,
    )

    return gp_samples
