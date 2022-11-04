import pandas as pd
import numpy as np


class Perceptron:

    # Initialization function for Epoch and Learning Rate
    def __init__(self):
        self.T = 10
        self.learning_rate = 0.1

    def standard_perceptron(self, x, y):
        n = x.shape[0]
        dim = x.shape[1]
        weight_vector = np.zeros(dim)

        # shuffle list of indices to reference instead of shuffling all data
        indices = np.arange(n)

        for epoch in range(self.T):
            # shuffle the data
            np.random.shuffle(indices)
            for i in indices:
                if np.sum(x[i] * weight_vector) * y[i] <= 0:  # if y(i) != y'
                    # W(t+1) <- w(t) + r(y(i)x(i))
                    weight_vector = weight_vector + self.learning_rate * y[i] * x[i]
        return weight_vector

    def voted_perceptron(self, x, y):
        n = x.shape[0]
        dim = x.shape[1]
        # initialize w to 0
        weight_vector = np.zeros(dim)
        # shuffle indices instead of data
        indices = np.arange(n)
        # initialize result values
        correct_list = np.array([])
        weight_list = np.array([])

        correct_predictions = 0
        for epoch in range(self.T):
            np.random.shuffle(indices)
            # for each x(i)y(i) in shuffled data
            for i in indices:
                if np.sum(x[i] * weight_vector) * y[i] <= 0:
                    # update lists and weight vectors on incorrect prediction, runs a new perceptron
                    weight_list = np.append(weight_list, weight_vector)
                    correct_list = np.append(correct_list, correct_predictions)
                    weight_vector = weight_vector + self.learning_rate * y[i] * x[i]
                    correct_predictions = 1
                else:
                    correct_predictions = correct_predictions + 1
        num = correct_list.shape[0]
        weight_list = np.reshape(weight_list, (num, -1))
        # return the number of correct predictions list with the correlating weight vector list
        return correct_list, weight_list

    def averaged_perceptron(self, x, y):
        n = x.shape[0]
        dim = x.shape[1]

        weight_vector = np.zeros(dim)
        a = np.zeros(dim)

        indices = np.arange(n)

        for epoch in range(self.T):
            np.random.shuffle(indices)
            for i in indices:
                if np.sum(x[i] * weight_vector) * y[i] <= 0:
                    weight_vector = weight_vector + self.learning_rate * y[i] * x[i]
                a = a + weight_vector
        return a
