import torch

from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

torch.set_default_dtype(torch.float64)


inputs = torch.load("normalized_inputs.txt")
objective1_vals = torch.load("normalized_objective1_vals.txt").unsqueeze(1)
objective2_vals = torch.load("normalized_timenormalized_torque_minimize.txt")
print(objective1_vals.shape)
print(objective2_vals.shape)
objective_vals = torch.cat([objective1_vals, objective2_vals], dim=1)
torch.save(objective_vals, "objective_vals.txt")
print(objective_vals.shape)
model = SingleTaskGP(
    train_X=inputs, train_Y=objective_vals, outcome_transform=Standardize(2)
)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)
state_dict = model.state_dict()
torch.save(state_dict, "exo2_surrogate_state_dict.json")
