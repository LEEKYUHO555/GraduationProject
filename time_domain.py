import numpy as np

######## STATUS AND CONSTANTS

EPOCH = 1
TIMESTAMP = 5
learning_rate = 0.01
is_DFA = True
is_Weight_clip = True

######## DATA INPUT

x_train_timestamp = np.fromfile('Data/train_data.dat', dtype=float)
x_train_timestamp = np.reshape(x_train_timestamp, (55000, 784, TIMESTAMP))
x_test_timestamp = np.fromfile('Data/test_data.dat', dtype=float)
x_test_timestamp = np.reshape(x_test_timestamp, (10000, 784, TIMESTAMP))

y_train = np.loadtxt('Data/train_label.txt', dtype=int)
y_test = np.loadtxt('Data/test_label.txt', dtype=int)

train_DATASIZE = np.size(y_train, axis=0)
test_DATASIZE = np.size(y_test, axis=0)

print('Input finished!')

######## PARAMETER INITIALIZE

weight1 = np.random.randn(784, 512) / np.sqrt(784)
bias1 = np.full(512, 0.00)  # can be slight positive biased for DEAD RELUs 원래 0.001
weight2 = np.random.randn(512, 10) / np.sqrt(512)
bias2 = np.full(10, 0.00)
back_weight = np.random.uniform(low=-1, high=1, size=(10, 512)) / np.sqrt(512/6)

######## USEFUL FUNCs

def act_relu(x):
    return np.add(np.absolute(x), x) / 2

def grad_relu(x):
    temp_zeros = np.zeros(np.size(x))
    return ~np.equal(temp_zeros, x)                          # 이거 일단 bool 로 나올텐데 괜찮을지 모르겠

def compare_threshold(B):
    A = np.empty_like(B)
    A[:] = B
    A.fill(0.8)                                               # Threshold = 1

    return np.int_(B > A)

######## TRAINING

tot_cnt = 0
acc_cnt = 0

print('Training starts!')

for k in range(EPOCH):

    for i in range(train_DATASIZE):                         #Data shape : [datasize, pixels, timestamp]
        inf2 = np.zeros(512)
        inf3 = np.zeros(10)

        pred = np.zeros(10)

        for j in range(TIMESTAMP):

            grad_w1 = np.zeros((784, 512))
            grad_w2 = np.zeros((512, 10))
            grad_b1 = np.zeros(512)
            grad_b2 = np.zeros(10)

            delta2 = np.zeros(512)
            delta3 = np.zeros(10)

            a1 = x_train_timestamp[i, :, j]

            i2 = np.matmul(a1, weight1) + bias1
            inf2 += i2
            inf2 = np.clip(inf2, a_min=0, a_max=None)

            a2 = compare_threshold(inf2)
            inf2 = np.subtract(inf2, a2)

            inf3 += np.matmul(a2, weight2) + bias2                 # 여기가 좀 더 심각한 이슈인듯?
            inf3 = np.clip(inf3, a_min = 0, a_max=None)

            a3 = compare_threshold(inf3)
            inf3 = np.subtract(inf3, a3)

            output = a3

            delta3 = - output + y_train[i]
            if is_DFA:
                delta2 = np.matmul(delta3, (back_weight))       # 여기 grad relu 어떻게?
            else:
                delta2 = np.multiply(np.matmul(delta3, np.transpose(weight2)), grad_relu(inf2))

            temp_grad_w2 = np.matmul(np.reshape(a2,(512,1)), np.reshape(delta3,(1,10)))
            temp_grad_w1 = np.matmul(np.reshape(a1,(784,1)), np.reshape(delta2,(1,512)))
            temp_grad_b2 = delta3
            temp_grad_b1 = delta2

            # if i == 10 and j == 0:
            #     np.savetxt("ExpData/grad_w1_10.csv", temp_grad_w1, delimiter=", ")
            #     np.savetxt("ExpData/weight2_10.csv", temp_grad_w2, delimiter=", ")
            #
            # if i == 1000 and j == 0:
            #     np.savetxt("ExpData/grad_w1_5000.csv", temp_grad_w1, delimiter=", ")
            #     np.savetxt("ExpData/weight2_5000.csv", temp_grad_w2, delimiter=", ")

            if is_Weight_clip:
                weight1 = np.clip(np.add(weight1, learning_rate * temp_grad_w1), a_min=-0.1, a_max=0.1)
                weight2 = np.clip(np.add(weight2, learning_rate * temp_grad_w2), a_min=-0.1, a_max=0.1)
            else:
                weight1 = np.add(weight1, learning_rate * temp_grad_w1)
                weight2 = np.add(weight2, learning_rate * temp_grad_w2)

            bias1 = np.add(bias1, learning_rate * temp_grad_b1)
            bias2 = np.add(bias2, learning_rate * temp_grad_b2)

            pred = np.add(pred, output)

        final_result = np.argmax(pred)
        true_label = np.argmax(y_train[i])
        acc_cnt += np.equal(final_result, true_label)
        tot_cnt += 1

        if i % 1000 == 0:
            print('Epoch ' + str(k) + ', iter ' + str(i) + ': ' + str(acc_cnt/tot_cnt))
            acc_cnt = 0
            tot_cnt = 0

        if i == 5000:
            learning_rate /= 10

        if i == 15000:
            learning_rate /= 10

        ######## TEST
        if(i % 1000 == 0):
            test_acc_cnt = 0
            test_tot_cnt = 0

            for a in range(test_DATASIZE):
                test_inf2 = np.zeros(512)
                test_inf3 = np.zeros(10)

                test_pred = np.zeros(10)

                for b in range(TIMESTAMP):

                    test_a1 = x_test_timestamp[a, :, b]

                    test_i2 = np.matmul(test_a1, weight1) + bias1
                    test_inf2 += test_i2
                    test_inf2 = np.clip(test_inf2, a_min=0, a_max=None)

                    test_a2 = compare_threshold(test_inf2)
                    test_inf2 = np.subtract(test_inf2, test_a2)

                    test_inf3 += np.matmul(test_a2, weight2) + bias2  # 여기가 좀 더 심각한 이슈인듯?
                    test_inf3 = np.clip(test_inf3, a_min=0, a_max=None)

                    test_a3 = compare_threshold(test_inf3)
                    test_inf3 = np.subtract(test_inf3, test_a3)

                    test_output = test_a3

                    test_pred = np.add(test_pred, test_output)

                test_final_result = np.argmax(test_pred)
                test_true_label = np.argmax(y_test[a])
                test_acc_cnt += np.equal(test_final_result, test_true_label)
                test_tot_cnt += 1

            print('test : ' + str(test_acc_cnt / test_tot_cnt))





