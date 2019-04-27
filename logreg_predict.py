import pandas as pd
import numpy as np
import math

thetas = np.loadtxt('thetas.csv', delimiter=',')

thetaG = thetas[0]
thetaS = thetas[1]
thetaH = thetas[2]
thetaR = thetas[3]

def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

def predict(X, theta):
    res = sigmoid(np.dot(X, theta.T))
    return(res)

test = pd.read_csv("dataset_test.csv")

ids = test['Index']


test = test.drop(columns = ['Hogwarts House'])

test = test.drop(columns = ['First Name', 'Last Name', 'Birthday', 'Best Hand', 'Index'])
test = test.drop(columns = ['Arithmancy', 'Care of Magical Creatures', 'Astronomy'])
test = np.column_stack((test['Herbology'] ,test['Defense Against the Dark Arts']))
# test = test.dropna() #aulieu de dropna il faut fire la tang

test = np.c_[np.ones(test.shape[0]), test]


R = predict(test, thetaR)
G = predict(test, thetaG)
S = predict(test, thetaS)
H = predict(test, thetaH)

result = np.array([["Index", "Hogwarts House"]])
for i in range(len(R)):
	is_r = True if R[i] >= G[i] and R[i] >= S[i] and R[i] >= H[i] else False
	is_g = True if G[i] >= R[i] and G[i] >= S[i] and G[i] >= H[i] else False
	is_s = True if S[i] >= R[i] and S[i] >= G[i] and S[i] >= H[i] else False
	is_h = True if H[i] >= R[i] and H[i] >= G[i] and H[i] >= S[i] else False

	if is_r :
		house = "Ravenclaw"
	elif is_g :
		house = "Gryffindor"
	elif is_s :
		house = "Slytherin"
	else :
		house = "Hufflepuff"
	result = np.append (result, [[ids[i], house]], axis = 0)

np.savetxt('houses.csv', result, delimiter=',', fmt="%s")
