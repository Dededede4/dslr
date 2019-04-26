import pandas as pd
import numpy as np
#from matplotlib import pyplot as plt
import math;
#sns.set(style="ticks", color_codes=True)
df = pd.read_csv('dataset_train.csv')

data = pd.read_csv("dataset_train.csv")
data = data.drop(columns = ['First Name', 'Last Name', 'Birthday', 'Best Hand'])
data = data.drop(columns = ['Arithmancy', 'Care of Magical Creatures', 'Astronomy'])
data = data.dropna()

def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

def predict(X, theta):
    res = sigmoid(np.dot(X, theta.T))
    return(res)

def cost(X, y, theta):
    return((-1 / X.shape[0]) * np.sum(y * np.log(predict(X, theta)) + (1 - y) * np.log(1 - predict(X, theta))))

def fit(X, y, theta, alpha, num_iters):
    m = X.shape[0]
    J_history = []
    for _ in range(num_iters):
        theta = theta - (alpha / m) * (np.dot(predict(X, theta) - y, X))
        J_history.append(cost(X, y, theta))
    return theta, J_history


def normalise(x):
    return (x - np.mean(x)) / np.std(x)

X = data.values[:, 7:9]
X = np.column_stack((data['Herbology'] ,data['Defense Against the Dark Arts']))
y = np.column_stack(data['Hogwarts House'])
X = normalise(X)
X = np.c_[np.ones(X.shape[0]), X]

mask_Ravenclaw = y == "Ravenclaw"
mask_Gryffindor = y == "Gryffindor"
mask_Hufflepuff = y == "Hufflepuff"
mask_Slytherin = y == "Slytherin"
theta = np.zeros(3)

theta = np.zeros(3, dtype=float)
print (X.shape, theta.shape, y.shape)
theta, J_history = fit(X, mask_Ravenclaw, theta, 0.001, 600)

#fig = plt.figure()
#ax = plt.axes()
#ax.plot(J_history)


p = [1, 5.727180, 4.878861]
p = predict(p, theta)
print (p)

a = [1, -5.987446, 5.520605]
a = predict(a, theta)
print (a)