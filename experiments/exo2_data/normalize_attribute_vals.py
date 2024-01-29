import numpy as np
import torch
import csv

#file_name = "attribute_vals_maximize.txt"
#file_name = "timenormalized_torque_minimize.txt"
file_name = "inputs.txt"

with open(file_name, 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    data = [[float(item) for item in row] for row in data]

attribute_vals = np.array(data)
#print(attribute_vals[0])
print(attribute_vals.min(0))
normalized_attribute_vals = (attribute_vals - attribute_vals.min(0)) / (
    attribute_vals.max(0) - attribute_vals.min(0)
)
normalized_attribute_vals = torch.tensor(normalized_attribute_vals)
torch.save(normalized_attribute_vals, "normalized_" + file_name)
