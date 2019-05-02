import numpy as np

TIMESTAMP = 5
#
# x_train = np.array([[0.2, 1, 0.8], [0, 0.6, 0.5]])
# x_train_timestamp = np.zeros((2, 3, TIMESTAMP))
#
# for i in range(2):
#     for j in range(3):
#         prob = x_train[i][j]
#         x_train_timestamp[i][j] = np.random.choice([0, 1], size=5, p=[1 - prob, prob])
#
# print(x_train_timestamp)
#
# x_train_timestamp.tofile('test.dat')

b = np.loadtxt('train_label.txt', dtype=int)
print(np.shape(b))

# c = np.fromfile('test.dat', dtype=float)
# c = np.reshape(c, (2,3,TIMESTAMP))
# print(c)