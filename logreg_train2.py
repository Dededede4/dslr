import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})

data = pd.read_csv('dataset_train.csv')
data = data.dropna()

X = data.values[:, 6:8]
y = data.values[:, 1:2]
print(X.shape)
print(type(X))
print(y.shape)

theta = np.zeros(3)

def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

print(sigmoid(0.000))
print(sigmoid(30.0004))
print(sigmoid(-100.005))

def predict(X, theta):
    z = np.array(np.dot(X, theta),dtype=np.float32)
    return (sigmoid(z))

def cost(X, y, theta):
    return((-1 / X.shape[0]) * np.sum(y * np.log(predict(X, theta)) + (1 - y) * np.log(1 - predict(X, theta))))

X = np.c_[np.ones(X.shape[0]), X]
print (X)

theta = np.zeros(3, dtype=float)
print (predict(X, theta))

mask_Ravenclaw = y == "Ravenclaw"
mask_Gryffindor = y == "Gryffindor"
mask_Hufflepuff = y == "Hufflepuff"
mask_Slytherin = y == "Slytherin"

print ("==>", mask_Ravenclaw, "==>", mask_Gryffindor, "==>", mask_Hufflepuff, "==>", mask_Slytherin)

print (cost(X, mask_Ravenclaw, theta))
print (cost(X, mask_Gryffindor, theta))
print (cost(X, mask_Hufflepuff, theta))
print (cost(X, mask_Slytherin, theta))

def fit(X, y, theta, alpha, num_iters):
    m = X.shape[0]
    J_history = []
    for _ in range(num_iters):
        theta = theta - (alpha / m) * (np.dot(predict(X, theta) - y, X))
        J_history.append(cost(X, y, theta))
    return theta, J_history

theta = np.zeros(3, dtype=float)
print (X.shape, theta.shape, mask_Ravenclaw.shape)
print (theta, J_history = fit(X, mask_Ravenclaw, theta, 0.001, 1000))

#defense contre les force du mal
#