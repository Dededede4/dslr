import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})

data = pd.read_csv('dataset_train.csv')
data = data.dropna()

X = data.values[:, 6:8]
y = data.values[:, 1:2]
print(X.shape)
print(y.shape)

theta = np.zeros(3)

def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

print(sigmoid(0.000))
print(sigmoid(30.0004))
print(sigmoid(-100.005))

def predict(X, theta):
	z=np.array(np.dot(X, theta),dtype=np.float32)
	return(sigmoid(z))

def cost(X, y, theta):
    return((-1 / X.shape[0]) * np.sum(y * np.log(predict(X, theta)) + (1 - y) * np.log(1 - predict(X, theta))))

X = np.c_[np.ones(X.shape[0]), X]
print (X)

theta = np.zeros(3, dtype=float)
predict(X, theta)