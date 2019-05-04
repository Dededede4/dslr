import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})

data = pd.read_csv("dataset_train.csv")
data = data.drop(columns = ['First Name', 'Last Name', 'Birthday', 'Best Hand', 'Index'])
data = data.drop(columns = ['Arithmancy', 'Care of Magical Creatures', 'Astronomy'])
# data = data.drop(columns = ['Divination', 'Muggle Studies', 'History of Magic'])
# data = data.drop(columns = ['Transfiguration', 'Potions', 'Flying', 'Charms', 'Ancient Runes'])
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
    for _ in tqdm(range(num_iters)):
        theta = theta - (alpha / m) * (np.dot(predict(X, theta) - y, X))
        J_history.append(cost(X, y, theta))
    return theta, J_history

def normalise(x):
    return (x - np.mean(x)) / np.std(x)

# X = np.column_stack((data['Herbology'] ,data['Defense Against the Dark Arts']))
X = data.copy()
X = X.drop(columns = ['Hogwarts House'])
y = np.column_stack(data['Hogwarts House'])
X = normalise(X)
X = np.c_[np.ones(X.shape[0]), X]

mask_Ravenclaw = y == "Ravenclaw"
mask_Gryffindor = y == "Gryffindor"
mask_Hufflepuff = y == "Hufflepuff"
mask_Slytherin = y == "Slytherin"

theta = np.zeros(X.shape[1])

theta = np.zeros(X.shape[1], dtype=float)
thetaR, J_historyR = fit(X, mask_Ravenclaw, theta, 0.001, 500)
theta = np.zeros(X.shape[1], dtype=float)
thetaG, J_historyG = fit(X, mask_Gryffindor, theta, 0.001, 500)
theta = np.zeros(X.shape[1], dtype=float)
thetaH, J_historyH = fit(X, mask_Hufflepuff, theta, 0.001, 500)
theta = np.zeros(X.shape[1], dtype=float)
thetaS, J_historyS = fit(X, mask_Slytherin, theta, 0.001, 500)

thetas = np.array([thetaG[X.shape[0]-1], thetaS[X.shape[0]-1], thetaH[X.shape[0]-1], thetaR[X.shape[0]-1]])
np.savetxt('thetas.csv', thetas, delimiter=',')


fig = plt.figure()
ax = plt.axes()
ax.set_xlim([0, 500])
ax.set_ylim([1060, 1067])
ax.scatter([0, 10], [0, 10])
line_x = np.linspace(0,22.5, 20)
line_y = np.linspace(0,22.5, 20)
ax.plot(J_historyR)
ax.plot(J_historyG)
ax.plot(J_historyH)
ax.plot(J_historyS)
plt.show()