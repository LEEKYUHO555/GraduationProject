######## GOAL OF CODE
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(512, activation=tf.nn.relu),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10, activation=tf.nn.softmax)
# ])
#
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(x_train, y_train, epochs=5)
# model.evaluate(x_test, y_test)


import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

######## STATUS AND CONSTANTS

is_train = True
EPOCH = 30
batch_size = 100
learning_rate = 0.5

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

######## DATA INPUT

# mnist = tf.keras.datasets.mnist
#
# (x_train, y_train),(x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0           # 0과 1 사이의 값으로 normalize
#
# x_train = np.reshape(x_train, (60000, 28*28))
# x_test = np.reshape(x_test, (10000, 28*28))                 # IS THIS RIGHT? MAYBE OK
#
# nb_classes = 10                                             # ONE-HOT ENCODING
# y_train = np.eye(nb_classes)[y_train]
# y_test = np.eye(nb_classes)[y_test]

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)      # already normalized
x_train = mnist.train.images
y_train = mnist.train.labels

x_test = mnist.test.images
y_test = mnist.test.labels

######## PARAMETER INITIALIZE

weight1 = np.random.randn(784, 512) / np.sqrt(784 / 2)      # He initialization
bias1 = np.full(512, 0.00)                                  # can be slight positive biased for DEAD RELUs 원래 0.001
weight2 = np.random.randn(512, 10) / np.sqrt(512 / 2)
bias2 = np.full(10, 0.00)
back_weight = np.random.randn(10,512)/np.sqrt(512/2)

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
            a2 = act_relu(z2)
            z3 = np.matmul(a2, weight2) + np.tile(bias2, (batch_size,1))

            output = softmax(z3)

            # cross entropy error related code fix if needed

            delta3 = - output + y_train[j * batch_size : (j + 1) * batch_size]
            #delta2 = np.multiply(np.matmul(delta3, np.transpose(weight2)), grad_relu(a2))  # sus1 여기 grad relu

            delta2 = np.multiply( np.matmul(delta3, (back_weight)), grad_relu(a2) )     # sus1 여기 grad relu



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

            # for k in range(batch_size):                     # 이렇게 계산하면 연산 낭비 씹오짐.. 가능하면 행렬연산으로 바꾸기 나중에
            #
            #     idx = j * batch_size + k
            #     delta2 = np.zeros(512)
            #     delta3 = np.zeros(10)
            #
            #     node2 = np.matmul(x_train[idx], weight1) + bias1                    # feed forward
            #     act_node2 = act_relu(node2)
            #     node3 = np.matmul(act_node2, weight2) + bias2
            #     output = softmax(node3)
            #
            #     cross_entropy_error = cross_entropy(output, y_train[idx])
            #
            #     delta3 = output - y_train
            #     delta2 = np.multiply( np.matmul(delta3, np.transpose(weight2)), grad_relu(act_node2) )
            #
            #     temp_grad_w2 = np.matmul( np.transpose(act_node2), delta3 )
            #     temp_grad_w1 = np.matmul( np.transpose(x_train[idx]), delta2 )
            #     temp_grad_b2 = delta3
            #     temp_grad_b1 = delta2
            #
            #     grad_w1 += temp_grad_w1
            #     grad_w2 += temp_grad_w2
            #     grad_b1 += temp_grad_b1
            #     grad_b2 += temp_grad_b2
            #
            # grad_w1 = learning_rate * grad_w1 / batch_size  # 여기 배치사이즈로 나눠주는거 맞겠지?
            # grad_w2 = learning_rate * grad_w2 / batch_size
            # grad_b1 = learning_rate * grad_b1 / batch_size
            # grad_b2 = learning_rate * grad_b2 / batch_size
            #
            # weight1 += grad_w1
            # weight2 += grad_w2
            # bias1 += grad_b1
            # bias2 += grad_b2

        ## TEST CODE

        test_node2 = np.matmul(x_test, weight1) + np.tile(bias1, (10000,1))  # x_test.size(axis=0) 해야하는데 안되서 1만 처넣음
        test_act_node2 = act_relu(test_node2)
        test_node3 = np.matmul(test_act_node2, weight2) + np.tile(bias2, (10000 ,1))
        test_output = softmax(test_node3)

        pred = np.argmax(test_output, axis=1)
        true_label = np.argmax(y_test, axis=1)
        accuracy = np.sum(np.equal(pred, true_label)) / np.size(true_label, axis=0)

        print('Epoch '+ str(i+1) + ' accuracy : ' + str(accuracy))



