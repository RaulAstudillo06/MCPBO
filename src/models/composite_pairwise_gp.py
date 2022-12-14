import torch

from botorch.fit import fit_gpytorch_mll
from botorch.models.likelihoods import PairwiseLogitLikelihood
from botorch.models.model import Model
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.posteriors import Posterior
from torch import Tensor

from src.models.pairwise_kernel_variational_gp import PairwiseKernelVariationalGP
from src.utils import training_data_for_pairwise_gp


class CompositePairwiseGP(Model):
    r""" """

    def __init__(
        self,
        queries,
        responses,
        use_attribute_uncertainty=True,
        fit_model_flag=True,
    ) -> None:
        r""" """
        self.queries = queries
        self.responses = responses
        self.use_attribute_uncertainty = use_attribute_uncertainty

        output_dim = responses.shape[-1] - 1
        attribute_models = []
        attribute_means = []
        lower_bounds = []
        upper_bounds = []
        for j in range(output_dim):
            attribute_model = PairwiseKernelVariationalGP(queries, responses[..., j], fit_aux_model_flag=fit_model_flag)
            attribute_mean = attribute_model(queries).mean
            if self.use_attribute_uncertainty:
                attribute_std = torch.sqrt(attribute_model(queries).variance)
                lower_bounds.append((attribute_mean - attribute_std).min().item())
                upper_bounds.append((attribute_mean + attribute_std).max().item())
            else:
                lower_bounds.append(attribute_mean.min().item())
                upper_bounds.append(attribute_mean.max().item())
            attribute_models.append(attribute_model)
            attribute_means.append(attribute_mean.detach())
        self.lower_bounds = torch.as_tensor(lower_bounds).to(
            device=queries.device, dtype=queries.dtype
        )
        self.upper_bounds = torch.as_tensor(upper_bounds).to(
            device=queries.device, dtype=queries.dtype
        )
        self.attribute_models = attribute_models
        attribute_means = torch.cat(attribute_means, dim=-1)

        utility_queries = (attribute_means - self.lower_bounds) / (
            self.upper_bounds - self.lower_bounds
        )
        utility_model = PairwiseKernelVariationalGP(utility_queries, responses[..., -1], fit_aux_model_flag=fit_model_flag)
        self.utility_model = [utility_model]

    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model."""
        return 1

    def posterior(self, X: Tensor, observation_noise=False, posterior_transform=None):
        r"""Computes the posterior over model outputs at the provided points.
        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.
            observation_noise: If True, add the observation noise from the
                likelihood to the posterior. If a Tensor, use it directly as the
                observation noise (must be of shape `(batch_shape) x q`).
        Returns:
            A `GPyTorchPosterior` object, representing a batch of `b` joint
            distributions over `q` points. Includes observation noise if
            specified.
        """
        return MultivariateNormalComposition(
            attribute_models=self.attribute_models,
            utility_model=self.utility_model,
            X=X,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
            use_attribute_uncertainty=self.use_attribute_uncertainty,
        )

    def forward(self, x: Tensor):
        return MultivariateNormalComposition(
            attribute_model=self.attribute_models,
            utility_model=self.utility_model,
            X=x,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
            use_attribute_uncertainty=self.use_attribute_uncertainty,
        )


class MultivariateNormalComposition(Posterior):
    def __init__(
        self,
        attribute_models,
        utility_model,
        X,
        lower_bounds,
        upper_bounds,
        use_attribute_uncertainty=True,
    ):
        self.attribute_models = attribute_models
        self.utility_model = utility_model
        self.X = X
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.use_attribute_uncertainty = use_attribute_uncertainty

    @property
    def device(self) -> torch.device:
        r"""The torch device of the posterior."""
        return "cpu"

    @property
    def dtype(self) -> torch.dtype:
        r"""The torch dtype of the posterior."""
        return torch.double

    @property
    def event_shape(self) -> torch.Size:
        r"""The event shape (i.e. the shape of a single sample) of the posterior."""
        shape = list(self.X.shape)
        shape[-1] = len(self.attribute_models) + 1
        shape = torch.Size(shape)
        return shape

    def rsample(self, sample_shape=torch.Size(), base_samples=None):
        # t0 =  time.time()
        # print(base_samples.shape)
        attribute_samples = []
        for j, attribute_model in enumerate(self.attribute_models):
            attribute_multivariate_normal = attribute_model.posterior(self.X)
            if self.use_attribute_uncertainty:
                if base_samples is not None:
                    attribute_samples.append(
                        attribute_multivariate_normal.rsample(
                            sample_shape, base_samples=base_samples[..., [j]]
                        )  # [..., 0]
                    )
                else:
                    attribute_samples.append(
                        attribute_sample=attribute_multivariate_normal.rsample(
                            sample_shape
                        )  # [..., 0]
                    )
            else:
                attribute_samples.append(attribute_multivariate_normal.mean)
        attribute_samples = torch.cat(attribute_samples, dim=-1)
        normalized_attribute_samples = (attribute_samples - self.lower_bounds) / (
            self.upper_bounds - self.lower_bounds
        )
        utility_multivariate_normal = self.utility_model[0].posterior(
            normalized_attribute_samples
        )
        if base_samples is not None:
            utility_samples = utility_multivariate_normal.mean + torch.mul(
                torch.sqrt(utility_multivariate_normal.variance),
                base_samples[..., [-1]],
            )
            # print(utility_multivariate_normal.mean.shape)
            # print(base_samples[..., [-1]].shape)
        else:
            utility_samples = utility_multivariate_normal.rsample(sample_shape)
        return utility_samples
