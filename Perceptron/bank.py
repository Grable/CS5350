import pandas as pd
import numpy as np
import Perceptron

train_data = pd.read_csv("../CS5350/Perceptron/train.csv", header=None)

# process data
raw = train_data.values
train_x = np.copy(raw)
rows = raw.shape[0]
columns = raw.shape[1]
train_x[:, columns - 1] = 1
train_y = raw[:, columns - 1]
train_y = 2 * train_y - 1

test_data = pd.read_csv("../CS5350/Perceptron/test.csv", header=None)

raw = test_data.values
test_x = np.copy(raw)
test_rows = raw.shape[0]
test_columns = raw.shape[1]
test_x[:, test_columns - 1] = 1
test_y = raw[:, test_columns - 1]
test_y = 2 * test_y - 1

bank = Perceptron.Perceptron()


# standard perceptron
w = bank.standard_perceptron(train_x, train_y)
print(w)
w = np.reshape(w, (-1, 1))
pred = np.matmul(test_x, w)
pred[pred > 0] = 1
pred[pred <= 0] = -1
err = np.sum(np.abs(pred - np.reshape(test_y, (-1, 1)))) / 2 / test_y.shape[0]
print("standard perceptron error: ", err)

#######################################################################################

# voted perceptron
correct_list, weight_list = bank.voted_perceptron(train_x, train_y)
print(weight_list)
weight_list = np.transpose(weight_list)
pred = np.matmul(test_x, weight_list)
pred[pred > 0] = 1
pred[pred <= 0] = -1

correct_list = np.reshape(correct_list, (-1, 1))
voted = np.matmul(pred, correct_list)
voted[voted > 0] = 1
voted[voted <= 0] = -1
error = np.sum(np.abs(voted - np.reshape(test_y, (-1, 1)))) / 2 / test_y.shape[0]
print("voted perceptron error: ", error)
print(correct_list)

#######################################################################################

# averaged perceptron
w = bank.averaged_perceptron(train_x, train_y)
w = np.reshape(w, (-1, 1))
pred = np.matmul(test_x, w)
pred[pred > 0] = 1
pred[pred <= 0] = -1
err = np.sum(np.abs(pred - np.reshape(test_y, (-1, 1)))) / 2 / test_y.shape[0]
print("averaged perceptron error: ", err)
print(w)
