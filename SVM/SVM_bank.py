import pandas as pd
import numpy as np
import SVM

train_data = pd.read_csv("../CS5350/SVM/train.csv", header=None)
# process data
raw = train_data.values
num_col = raw.shape[1]
num_row = raw.shape[0]
train_x = np.copy(raw)
train_x[:, num_col - 1] = 1
train_y = raw[:, num_col - 1]
train_y = 2 * train_y - 1

test_data = pd.read_csv("../CS5350/SVM/test.csv", header=None)
raw = test_data.values
num_col = raw.shape[1]
num_row = raw.shape[0]
test_x = np.copy(raw)
test_x[:, num_col - 1] = 1
test_y = raw[:, num_col - 1]
test_y = 2 * test_y - 1

C_set = np.array([100, 500, 700])
C_set = C_set / 873
gammas = np.array([0.1, 0.5, 1, 5, 100])

svm = SVM.SVM()
for C in C_set:
    print("C: ", C)
    svm.set_C(C)
    w = svm.train_primal(train_x, train_y)
    w = np.reshape(w, (5, 1))

    prediction = np.matmul(train_x, w)
    prediction[prediction > 0] = 1
    prediction[prediction <= 0] = -1
    trainError = (
        np.sum(np.abs(prediction - np.reshape(train_y, (-1, 1)))) / 2 / train_y.shape[0]
    )

    prediction = np.matmul(test_x, w)
    prediction[prediction > 0] = 1
    prediction[prediction <= 0] = -1

    testError = (
        np.sum(np.abs(prediction - np.reshape(test_y, (-1, 1)))) / 2 / test_y.shape[0]
    )
    print(
        "linear SVM Primal training error: ", trainError, " testing error: ", testError
    )
    w = np.reshape(w, (1, -1))
    print("Primal: ", w)

    # dual form
    w = svm.train_dual(train_x[:, [x for x in range(num_col - 1)]], train_y)
    print("Dual: ", w)

    w = np.reshape(w, (5, 1))

    prediction = np.matmul(train_x, w)
    prediction[prediction > 0] = 1
    prediction[prediction <= 0] = -1
    trainError = (
        np.sum(np.abs(prediction - np.reshape(train_y, (-1, 1)))) / 2 / train_y.shape[0]
    )

    prediction = np.matmul(test_x, w)
    prediction[prediction > 0] = 1
    prediction[prediction <= 0] = -1

    testError = (
        np.sum(np.abs(prediction - np.reshape(test_y, (-1, 1)))) / 2 / test_y.shape[0]
    )
    print("linear SVM Dual training error: ", trainError, " testing error: ", testError)

    # gaussian kernel
    c = 0
    for gamma in gammas:
        print("gamma: ", gamma)
        svm.set_gamma(gamma)
        alpha = svm.train_gaussian_kernel(
            train_x[:, [x for x in range(num_col - 1)]], train_y
        )
        idx = np.where(alpha > 0)[0]
        print("#sv: ", len(idx))
        # train
        y = svm.predict_gaussian_kernel(
            alpha,
            train_x[:, [x for x in range(num_col - 1)]],
            train_y,
            train_x[:, [x for x in range(num_col - 1)]],
        )
        trainError = (
            np.sum(np.abs(y - np.reshape(train_y, (-1, 1)))) / 2 / train_y.shape[0]
        )

        # test
        y = svm.predict_gaussian_kernel(
            alpha,
            train_x[:, [x for x in range(num_col - 1)]],
            train_y,
            test_x[:, [x for x in range(num_col - 1)]],
        )
        testError = (
            np.sum(np.abs(y - np.reshape(test_y, (-1, 1)))) / 2 / test_y.shape[0]
        )
        print(
            "nonlinear SVM training error: ", trainError, " testing error: ", testError
        )

        if c > 0:
            intersect = len(np.intersect1d(idx, old_idx))
            print("# of intersecting Support Vectors: ", intersect)
        c = c + 1
        old_idx = idx
