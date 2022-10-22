import pandas as pd
import numpy as np


class LMS:
    # method: 0 batch gradient descent
    #         1 SGD
    #         2 normal equation
    def __init__(self):
        self.learning_rate = 0.01
        self.threshold = 10e-6
        self.max_iterations = 1000

    def set_learning_rate(self, lr):
        self.learning_rate = lr

    def set_threshold(self, threshold):
        self.threshold = threshold

    def set_max_iter(self, max_iter):
        self.max_iterations = max_iter

    # x is augmented
    def optimize_BGD(self, x, y):
        dim = x.shape[1]
        # update difference
        diff = 1
        # init w
        w = np.zeros([dim, 1])
        cost_function_vals = []
        i = 0
        while diff > self.threshold and i < self.max_iterations:
            i = i + 1
            tmp = np.reshape(np.squeeze(np.matmul(x, w)) - y, (-1, 1))
            # Adjust delta value
            g = np.reshape(np.sum(np.transpose(tmp * x), axis=1), (-1, 1))
            delta = -self.learning_rate * g

            # Change weight vector
            w_new = w + delta
            diff = np.sqrt(np.sum(np.square(delta)))
            w = w_new

            # Get the Cost
            tmp = np.reshape(np.squeeze(np.matmul(x, w)) - y, (-1, 1))
            val = 0.5 * np.sum(np.square(tmp))
            cost_function_vals.append(val)
        return w

    def optimize_SGD(self, x, y):
        dim = x.shape[1]
        n = x.shape[0]
        # init w
        w = np.zeros([dim, 1])
        val = 1
        cost_function_vals = []
        i = 0
        while val > self.threshold:
            i = i + 1
            # random index
            idx = np.random.randint(n, size=1)
            x1 = x[idx]
            y1 = y[idx]

            # Adjust delta and weight
            g = np.sum(np.transpose((np.matmul(x1, w) - y1) * x1), axis=1)
            delta = -self.learning_rate * np.reshape(g, (-1, 1))
            w_new = w + delta
            w = w_new
            # Get the cost
            tmp = np.reshape(np.squeeze(np.matmul(x, w)) - y, (-1, 1))
            val = 0.5 * np.sum(np.square(tmp))
            cost_function_vals.append(val)
        return w
