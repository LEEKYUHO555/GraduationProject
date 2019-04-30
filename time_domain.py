import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

######## STATUS AND CONSTANTS

EPOCH = 30
TIMESTAMP = 5
learning_rate = 0.001

######## DATA INPUT

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)      # already normalized

x_train = mnist.train.images
x_test = mnist.test.images

y_train = mnist.train.labels
y_test = mnist.test.labels

x_train = x_train[0:300]
y_train = y_train[0:300]

x_test = x_test[0:100]
y_test = y_test[0:100]

######## DATA TO TIME-DOMAIN

train_DATASIZE = np.size(x_train, axis=0)
test_DATASIZE = np.size(x_test, axis=0)

x_train_timestamp = np.zeros((train_DATASIZE, 784, TIMESTAMP))
x_test_timestamp = np.zeros((test_DATASIZE, 784, TIMESTAMP))

for i in range(train_DATASIZE):
    for j in range(784):
        prob = x_train[i][j]
        x_train_timestamp[i][j] = np.random.choice([0, 1], size=5, p=[1 - prob, prob])
    if i%100 ==0:
        print('train ' + str(i/train_DATASIZE))

# for i in range(test_DATASIZE):
#     for j in range(784):
#         prob = x_test[i][j]
#         x_test_timestamp[i][j] = np.random.choice([0, 1], size=5, p=[1 - prob, prob])

# y_train_timestamp = y_train[..., np.newaxis]
# y_train_timestamp = np.tile(y_train_timestamp,(1,1,5))
#
# y_test_timestamp = y_test[..., np.newaxis]
# y_test_timestamp = np.tile(y_test_timestamp,(1,1,5))

######## PARAMETER INITIALIZE

weight1 = np.random.randn(784, 512) / np.sqrt(784 / 2)  # He initialization
bias1 = np.full(512, 0.00)  # can be slight positive biased for DEAD RELUs 원래 0.001
weight2 = np.random.randn(512, 10) / np.sqrt(512 / 2)
bias2 = np.full(10, 0.00)
back_weight = np.random.uniform(low=-1, high=1, size=(10, 512)) / np.sqrt(512)

######## USEFUL FUNCs

def act_relu(x):
    return np.add(np.absolute(x), x) / 2

def grad_relu(x):
    temp_zeros = np.zeros(np.size(x))
    return ~np.equal(temp_zeros, x)                          # 이거 일단 bool 로 나올텐데 괜찮을지 모르겠

def compare_threshold(B):
    A = np.empty_like(B)
    A[:] = B
    A.fill(1)

    return np.int_(B > A)

######## TRAINING

tot_cnt = 0
acc_cnt = 0

for k in range(EPOCH):
    
    for i in range(train_DATASIZE):                         #Data shape : [datasize, pixels, timestamp]
        inf2 = np.zeros(512)
        inf3 = np.zeros(10)

        for j in range(TIMESTAMP):

            grad_w1 = np.zeros((784, 512))
            grad_w2 = np.zeros((512, 10))
            grad_b1 = np.zeros(512)
            grad_b2 = np.zeros(10)

            delta2 = np.zeros(512)
            delta3 = np.zeros(10)

            a1 = x_train_timestamp[i, :, j]

            i2 = act_relu(np.matmul(a1, weight1) + bias1)         # 여기 렐루 써도 되나?
            inf2 += i2

            a2 = compare_threshold(inf2)
            inf2 = np.subtract(inf2, a2)

            inf3 += np.matmul(a2, weight2) + bias2                 # 여기가 좀 더 심각한 이슈인듯?

            a3 = compare_threshold(inf3)
            inf3 = np.subtract(inf3, a3)

            output = a3

            delta3 = - output + y_train[i]
            delta2 = np.multiply(np.matmul(delta3, (back_weight)), grad_relu(i2))       # 여기 grad relu 어떻게?

            temp_grad_w2 = np.matmul(np.reshape(a2,(512,1)), np.reshape(delta3,(1,10)))
            temp_grad_w1 = np.matmul(np.reshape(a1,(784,1)), np.reshape(delta2,(1,512)))
            temp_grad_b2 = delta3
            temp_grad_b1 = delta2

            weight1 = np.add(weight1, learning_rate * temp_grad_w1)
            weight2 = np.add(weight2, learning_rate * temp_grad_w2)

            bias1 = np.add(bias1, learning_rate * temp_grad_b1)
            bias2 = np.add(bias2, learning_rate * temp_grad_b2)

            ## train accuracy
            pred = np.argmax(output)
            true_label = np.argmax(y_train[i])
            acc_cnt += np.equal(pred, true_label)
            tot_cnt += 1

        if i % 100 == 0:
            print(str(i) + ' ' + str(acc_cnt/tot_cnt))

            # ## TEST CODE
            # if j % 1000 == 0:
            #     test_node2 = np.matmul(x_test, weight1) + np.tile(bias1,
            #                                                       (10000, 1))  # x_test.size(axis=0) 해야하는데 안되서 1만 처넣음
            #     if is_quantize:
            #         test_act_node2 = stair_func(act_relu(test_node2))
            #     else:
            #         test_act_node2 = act_relu(test_node2)
            #     test_node3 = np.matmul(test_act_node2, weight2) + np.tile(bias2, (10000, 1))
            #     if is_quantize:
            #         test_output = stair_func(test_node3)
            #     else:
            #         test_output = test_node3
            #
            #     pred = np.argmax(test_output, axis=1)
            #     true_label = np.argmax(y_test, axis=1)
            #     accuracy = np.sum(np.equal(pred, true_label)) / np.size(true_label, axis=0)
            #
            #     # print('Epoch ' + str(i + 1) + ' iter ' + str(j) + ' accuracy : ' + str(accuracy))
            #     print( str(j) + ' ' + str(accuracy))



