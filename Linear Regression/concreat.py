import pandas as pd
import numpy as np
import LMS

train_data = pd.read_csv("../CS5350/Linear Regression/train.csv", header=None)

# training data values
raw = train_data.values
num_col = raw.shape[1]
num_row = raw.shape[0]
train_x = np.copy(raw)
train_x[:, num_col - 1] = 1
train_y = raw[:, num_col - 1]
train_y = 2 * train_y - 1

# testing data values
test_data = pd.read_csv("../CS5350/Linear Regression/test.csv", header=None)
raw = test_data.values
num_col = raw.shape[1]
num_row = raw.shape[0]
test_x = np.copy(raw)
test_x[:, num_col - 1] = 1
test_y = raw[:, num_col - 1]
test_y = 2 * test_y - 1

lms = LMS.LMS()

# Batch Gradient Descent
w = lms.optimize_BGD(train_x, train_y)
print("BGD w: ", w)

# Use W to get final cost value
tmp = np.reshape(np.squeeze(np.matmul(test_x, w)) - test_y, (-1, 1))
fv = 0.5 * np.sum(np.square(tmp))
print("BGD test_fv: ", fv)

# Stochastic Gradient Descent
# TAKES A LONG TIME!!!!
w = lms.optimize_SGD(train_x, train_y)
print("SGD w: ", w)
# Use W to get final cost value
tmp = np.reshape(np.squeeze(np.matmul(test_x, w)) - test_y, (-1, 1))
fv = 0.5 * np.sum(np.square(tmp))
print("SGD test_fv: ", fv)
