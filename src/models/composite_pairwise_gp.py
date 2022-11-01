import torch

from copy import deepcopy
from botorch.fit import fit_gpytorch_model
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.posteriors import Posterior
from torch import Tensor


from src.utils import training_data_for_pairwise_gp
from src.models.likelihoods.pairwise import PairwiseLogitLikelihood
from src.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood


class CompositePairwiseGP(Model):
    r""" """

    def __init__(
        self,
        queries,
        responses,
    ) -> None:
        r""" """
        self.queries = queries
        self.responses = responses

        output_dim = responses.shape[-1] - 1
        attribute_models = []
        attribute_means = []
        lower_bounds = []
        upper_bounds = []
        for j in range(output_dim):
            datapoints, comparisons = training_data_for_pairwise_gp(
                queries, responses[:, j]
            )
            attribute_model = PairwiseGP(
                datapoints,
                comparisons,
                likelihood=PairwiseLogitLikelihood(),
                jitter=1e-4,
            )

            mll = PairwiseLaplaceMarginalLogLikelihood(
                likelihood=attribute_model.likelihood, model=attribute_model
            )
            fit_gpytorch_model(mll)
            attribute_model = attribute_model.to(
                device=queries.device, dtype=queries.dtype
            )
            attribute_mean = attribute_model(queries).mean
            attribute_std = torch.sqrt(attribute_model(queries).variance)

            lower_bounds.append((attribute_mean - attribute_std).min().item())
            upper_bounds.append((attribute_mean + attribute_std).max().item())
            attribute_models.append(deepcopy(attribute_model))
            attribute_means.append(deepcopy(attribute_mean.detach().unsqueeze(-1)))

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
        datapoints, comparisons = training_data_for_pairwise_gp(
            utility_queries, responses[:, -1]
        )
        utility_model = PairwiseGP(
            datapoints,
            comparisons,
            likelihood=PairwiseLogitLikelihood(),
            jitter=1e-4,
        )

        mll = PairwiseLaplaceMarginalLogLikelihood(
            likelihood=utility_model.likelihood, model=utility_model
        )
        fit_gpytorch_model(mll)
        utility_model = utility_model.to(device=queries.device, dtype=queries.dtype)
        self.utility_model = [utility_model]

    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model."""
        return 1

    def posterior(self, X: Tensor, observation_noise=False):
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
        )

    def forward(self, x: Tensor):
        return MultivariateNormalComposition(
            attribute_model=self.attribute_models,
            utility_model=self.utility_model,
            X=x,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
        )


class MultivariateNormalComposition(Posterior):
    def __init__(
        self,
        attribute_models,
        utility_model,
        X,
        lower_bounds=None,
        upper_bounds=None,
    ):
        self.attribute_models = attribute_models
        self.utility_model = utility_model
        self.X = X
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

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
        # print(self.X.shape)
        attribute_samples = []
        for j, attribute_model in enumerate(self.attribute_models):
            attribute_multivariate_normal = attribute_model.posterior(self.X)
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
        attribute_samples = torch.cat(attribute_samples, dim=-1)
        # print(attribute_samples.shape)
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
        else:
            utility_samples = utility_multivariate_normal.rsample(sample_shape)
        return utility_samples
