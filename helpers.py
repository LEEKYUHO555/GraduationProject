import numpy as np

# bias1 = np.random.uniform(low=-1, high=1, size=(2, 3))
# print(bias1)
# bias2 = bias1[...,np.newaxis]
# print(bias2)
# bias3 = np.tile(bias2,(1,1,5))
# print(bias3)
# print(np.shape(bias3))

def compare_threshold(B):
    A = np.empty_like(B)
    A[:] = B
    A.fill(1)

    return np.int_(B > A)

matrix= np.array([[0.1,1.1],[0.2,1.3]])

print(matrix)
print(compare_threshold(matrix))