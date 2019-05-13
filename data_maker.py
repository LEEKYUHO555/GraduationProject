import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

TIMESTAMP = 5

######## DATA INPUT

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)      # already normalized

x_train = mnist.train.images
x_test = mnist.test.images

y_train = mnist.train.labels
y_test = mnist.test.labels

######## DATA TO TIME-DOMAIN

train_DATASIZE = np.size(x_train, axis=0)
test_DATASIZE = np.size(x_test, axis=0)

x_train_timestamp = np.zeros((train_DATASIZE, 784, TIMESTAMP))
x_test_timestamp = np.zeros((test_DATASIZE, 784, TIMESTAMP))

for i in range(train_DATASIZE):
    for j in range(784):
        prob = x_train[i][j]
        x_train_timestamp[i][j] = np.random.choice([0, 1], size=5, p=[1 - prob, prob])
    if i%1000 ==0:
        print('train ' + str(i/train_DATASIZE))

x_train_timestamp.tofile('Data/train_data.dat')
np.savetxt('Data/train_label.txt', y_train, fmt='%d')

for i in range(test_DATASIZE):
    for j in range(784):
        prob = x_test[i][j]
        x_test_timestamp[i][j] = np.random.choice([0, 1], size=5, p=[1 - prob, prob])
    if i%1000 ==0:
        print('test ' + str(i/test_DATASIZE))

x_test_timestamp.tofile('Data/test_data.dat')
np.savetxt('Data/test_label.txt', y_test, fmt='%d')