import torch
from matplotlib import pyplot as plt

objective1 = torch.load("normalized_objective1_vals.txt").numpy()
objective2 = torch.load("normalized_timenormalized_torque_minimize.txt").numpy()
plt.plot(objective1, objective2, "bo")
plt.show()

