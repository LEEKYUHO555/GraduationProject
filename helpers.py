import numpy as np

def weight_quantize(x):
    a1 = np.arange(-0.1, 0.101, 0.001)
    a2 = np.arange(-0.1, 0.102, 0.002)
    idx = np.searchsorted(a1,x)
    temp = 0.001 * idx - 0.1 - 0.001
    idx2 = np.searchsorted(a2,temp)
    return 0.002 * idx2 - 0.1

temp = np.array([[-0.2, -0.08132, 0.07642], [0.3, 0.002432, 0.000231]])
print(weight_quantize(temp))




# print(np.searchsorted([0,1,2,3,4], 1, side='left', ))
# print(np.searchsorted([0,1,2,3,4], 1, side='right', ))