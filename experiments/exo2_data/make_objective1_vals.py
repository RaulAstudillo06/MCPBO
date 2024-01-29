import numpy as np
import torch

normalized_attribute_vals = torch.load("normalized_attribute_vals_maximize.txt")[:, [1, 2, 5]]
objective1_vals = np.average(normalized_attribute_vals, axis=1)

normalized_objective1_vals = (objective1_vals - objective1_vals.min(0)) / (
    objective1_vals.max(0) - objective1_vals.min(0)
)
torch.save(torch.tensor(normalized_objective1_vals), "normalized_objective1_vals.txt")
