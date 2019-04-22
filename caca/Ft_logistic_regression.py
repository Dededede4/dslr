import numpy as np
import matplotlib.pyplot as plt
from random import randint
from Ft_array import *

class Ft_logistic_regression():

    """
    This class handle LogisticRegression for a given dataset, it detects how many features there is in the dataset
    and uppon calling gradient descent, set the variable raw_thetas to the result of n epochs of gradient descent on starting raw_thetas.
    All values in X are normalized so regression is faster.

    when we access j, we access a feature, when we access i, we access a sample.

    for example, accessing X[i,j] means accessing the jth feature of the ith sample.
    we only access data in y with i and data in thetas with j for clarity

    attributes:

    m is the number of samples in the dataset
    n is the number of features in the dataset

    raw_data is the data sent by user. it's never overwritten.
    raw_thetas are the starting thetas sent by user. it's overwritten at the end of a gradient_descent call


    X is the normalized data array of size m * n + 1. The first column of X is filled with 1s for quick hypothesis computation
    y is the values vector of size m * 1.

    thetas are the normalized thetas vector of size n + 1. They're updated at each epochs of the gradient_descent function
    """

    def __init__(self, data, epochs = 1000, learning_rate = 0.001):
        self.cost = []
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.raw_data = data
        self.m = self.raw_data.shape[0];
        self.n = self.raw_data.shape[1] - 1;
        self.__get_scaled_data()
        self.thetas = np.zeros(self.n + 1)

    def gradient_descent(self):
        for n in range (0, 20000):
            self.thetas = self.__gradient_descent_epoch()
            if (n % 10 == 0):
                cost = self.get_cost()
            if (cost < 0.1):
                break
        self.raw_thetas = np.empty(len(self.thetas))
        self.raw_thetas[0] = ft_mean(self.y)
        for j in range(1, self.n + 1):
            self.raw_thetas[j] = (self.thetas[j]) / (ft_max(self.raw_data[:, j - 1]) - ft_min(self.raw_data[:, j - 1]))
            self.raw_thetas[0] -= self.raw_thetas[j] * np.nanmean(self.raw_data[:, j - 1])

    def get_cost(self):
        cost = 0;
        for i in range (0, self.m):
            if not np.isnan(self.X[i]).any():
                cost += self.y[i] * np.log(self.__predict(i)) + (1 - self.y[i]) * np.log(1 - self.__predict(i))
        cost /= float(self.m)
        return -cost

    # Adds a column filled with 1 (So Theta0 * x0 = Theta0) and apply ft_minft_max normalization to the raw data
    def __get_scaled_data(self):
        self.X = np.empty(shape=(self.m, self.n + 1)) # create the data matrix of size m * n
        self.X[:, 0] = 1
        self.y = np.empty(shape=(self.m, 1))
        self.y = self.raw_data[:, self.raw_data.shape[1] - 1] # copy y values of rawdata to y vector
        # assign raw data to X matrix
        for j in range(0, self.n):
            self.X[:, j + 1] = self.raw_data[:, j]
        # normalize the raw data stored in X matrix using min max normalization
        for j in range(1, self.n + 1):
            self.X[:, j] = (self.X[:, j] - ft_min(self.raw_data[:, j - 1])) / (ft_max(self.raw_data[:, j - 1]) - ft_min(self.raw_data[:, j - 1]))

    def __get_scaled_thetas(self):
        self.thetas = np.empty(self.n + 1)
        self.thetas[0] = self.raw_thetas[len(self.raw_thetas) - 1]
        for j in range(0, self.n):
            self.thetas[j + 1] = self.raw_thetas[j + 1] * (ft_max(self.raw_data[:, j]) - ft_min(self.raw_data[:, j]))

    def __gradient_descent_epoch(self):
        new_thetas = np.zeros(self.n + 1)
        samples = list(range(100))
        for i in range(self.m):
            j = randint(1, self.m)
            if (j < 100):
                samples[j] = i
        for i in samples:
            delta = self.__predict(i) - self.y[i]
            if not np.isnan(self.X[i]).any():
                for j in range(self.n + 1):
                    new_thetas[j] += delta * self.X[i, j]
        new_thetas[:] = self.thetas[:] - (self.learning_rate / float(self.m)) * new_thetas[:]
        return new_thetas

    def __predict(self, i):
        h = self.__sigmoid(np.dot(self.thetas, self.X[i]))
        return h

    def __sigmoid(self, h):
        return 1 / (1 + np.exp(-h))
