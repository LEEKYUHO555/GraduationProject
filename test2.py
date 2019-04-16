
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

######## STATUS AND CONSTANTS

is_quantize = True
is_BP_quantize = True
is_DFA = True
is_weight_quantize = True
is_train = True
EPOCH = 30
batch_size = 100
learning_rate = 0.1       # batch size 1일땐 0.001 근처로 설정해야 하는 것 같음. batch size 100일 땐 0.1 로도 충분함
quantize_size = 20

######## USEFUL FUNCs

def act_relu(x):
    return np.add(np.absolute(x), x) / 2

def softmax(x):
    e_x = np.exp(x)
    sum_ex = np.sum(e_x, axis=1)
    sum_ex.transpose()
    sum_ex = np.array([sum_ex, ] * 10).transpose()          # 원래 10 대신 size(x,axis=1) 할랬는데 안됨
    return np.divide(e_x, sum_ex)

def cross_entropy(pred, true_label):
    log_pred = np.log(pred)
    loss = -np.sum(np.multiply(true_label,log_pred), axis=1)
    return loss

def grad_relu(x):
    temp_zeros = np.zeros((np.size(x,axis=0),np.size(x,axis=1)))
    return ~np.equal(temp_zeros, x)                          # 이거 일단 bool 로 나올텐데 괜찮을지 모르겠

def stair_func(x):
    stair_values = np.arange(quantize_size) / quantize_size
    inds = np.digitize(x, stair_values)
    inds = np.subtract(inds,1) / quantize_size
    return inds

def weight_quantize(x):                                     # [-0.1, 0.1] 0.002 간격으로 반올림하는 함수
    a1 = np.arange(-0.1, 0.101, 0.001)
    a2 = np.arange(-0.1, 0.102, 0.002)
    idx = np.searchsorted(a1,x)
    temp = 0.001 * idx - 0.1 - 0.001
    idx2 = np.searchsorted(a2,temp)
    return 0.002 * idx2 - 0.1

######## DATA INPUT

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)      # already normalized
if is_quantize :
    x_train = stair_func(mnist.train.images)
    x_test = stair_func(mnist.test.images)
else:
    x_train = mnist.train.images
    x_test = mnist.test.images

y_train = mnist.train.labels
y_test = mnist.test.labels

######## PARAMETER INITIALIZE

if is_weight_quantize:
    weight1 = weight_quantize(np.random.randn(784, 512) / np.sqrt(784 / 2))  # He initialization
    bias1 = np.full(512, 0.00)  # can be slight positive biased for DEAD RELUs 원래 0.001
    weight2 = weight_quantize(np.random.randn(512, 10) / np.sqrt(512 / 2))
    bias2 = np.full(10, 0.00)
    # back_weight = weight_quantize(np.random.randn(10, 512) / np.sqrt(512 / 2))
    back_weight = weight_quantize(np.random.uniform(low=-1, high=1, size=(10, 512)) / np.sqrt(512))
else:
    weight1 = np.random.randn(784, 512) / np.sqrt(784 / 2)      # He initialization
    bias1 = np.full(512, 0.00)                                  # can be slight positive biased for DEAD RELUs 원래 0.001
    weight2 = np.random.randn(512, 10) / np.sqrt(512 / 2)
    bias2 = np.full(10, 0.00)
    back_weight = weight_quantize(np.random.uniform(low=-1, high=1, size=(10, 512)) / np.sqrt(512))

######## INFERENCE

if ~is_train:
    # blah blah
    print('BLAHBLAH')

######## TRAINING

if is_train:

    batch_number = int(np.size(x_train, axis=0) / batch_size)
    print(batch_number)

    for i in range(EPOCH):                                  # 0-4
        for j in range(batch_number):                       # 0-599

            grad_w1 = np.zeros((784, 512))
            grad_w2 = np.zeros((512, 10))
            grad_b1 = np.zeros(512)
            grad_b2 = np.zeros(10)

            train_data = x_train[j * batch_size : (j + 1) * batch_size]
            delta2 = np.zeros((batch_size, 512))
            delta3 = np.zeros((batch_size,10))

            z2 = np.matmul(train_data, weight1) + np.tile(bias1, (batch_size,1))
            if is_quantize:
                a2 = stair_func(act_relu(z2))
            else:
                a2 = act_relu(z2)
            z3 = np.matmul(a2, weight2) + np.tile(bias2, (batch_size,1))

            if is_quantize:
                output = stair_func(z3)
            else:
                output = z3;

            # cross entropy error related code fix if needed

            delta3 = - output + y_train[j * batch_size : (j + 1) * batch_size]

            if is_BP_quantize:
                if is_DFA:
                    delta2 = np.around(np.multiply(np.matmul(delta3, (back_weight)), grad_relu(a2)),
                                       decimals=3)  # DFA version
                else:
                    delta2 = np.multiply(np.matmul(delta3, np.transpose(weight2)), grad_relu(a2))  # Normal BP version

            else:
                if is_DFA:
                    delta2 = np.multiply(np.matmul(delta3, (back_weight)), grad_relu(a2))  # DFA version
                else:
                    delta2 = np.multiply(np.matmul(delta3, np.transpose(weight2)), grad_relu(a2))  # Normal BP version

            temp_grad_w2 = np.matmul(np.transpose(a2), delta3)
            temp_grad_w1 = np.matmul(np.transpose(train_data), delta2)
            temp_grad_b2 = delta3
            temp_grad_b1 = delta2

            temp_grad_w1 /= batch_size
            temp_grad_w2 /= batch_size
            temp_grad_b1 = np.sum(temp_grad_b1, axis=0) / batch_size
            temp_grad_b2 = np.sum(temp_grad_b2, axis=0) / batch_size

            weight1 = np.add(weight1, learning_rate * temp_grad_w1)
            weight2 = np.add(weight2, learning_rate * temp_grad_w2)
            bias1 = np.add(bias1, learning_rate * temp_grad_b1)
            bias2 = np.add(bias2, learning_rate * temp_grad_b2)

        if is_weight_quantize:
            weight1 = weight_quantize(weight1)
            weight2 = weight_quantize(weight2)

        ## TEST CODE

        test_node2 = np.matmul(x_test, weight1) + np.tile(bias1, (10000,1))  # x_test.size(axis=0) 해야하는데 안되서 1만 처넣음
        if is_quantize:
            test_act_node2 = stair_func(act_relu(test_node2))
        else:
            test_act_node2 = act_relu(test_node2)
        test_node3 = np.matmul(test_act_node2, weight2) + np.tile(bias2, (10000 ,1))
        if is_quantize:
            test_output = stair_func(test_node3)
        else:
            test_output = test_node3

        pred = np.argmax(test_output, axis=1)
        true_label = np.argmax(y_test, axis=1)
        accuracy = np.sum(np.equal(pred, true_label)) / np.size(true_label, axis=0)

        print('Epoch '+ str(i+1) + ' accuracy : ' + str(accuracy))



