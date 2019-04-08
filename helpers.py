import numpy as np

quantize_size = 10

x = np.array([[0.21, 0.42, 0.03, 0.16],[0.62, 0.64, 0.01, 0.716]])
stair_values = np.arange(quantize_size) / quantize_size
inds = np.digitize(x, stair_values)
inds = np.subtract(inds,1) / quantize_size
print(inds)