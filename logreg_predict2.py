import pandas as pd
import numpy as np
import math

thetas = np.loadtxt('thetas.csv', delimiter=',')

thetaG = thetas[0]
thetaS = thetas[1]
thetaH = thetas[2]
thetaR = thetas[3]

g = 84
s = 279
g2 = 142
g3 = 176

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
test = test.drop(columns = ['Divination', 'Muggle Studies', 'History of Magic'])
test = test.drop(columns = ['Transfiguration', 'Potions', 'Flying'])

col_mean = np.nanstd(test['Herbology'], axis=0)
# print (col_mean)
test['Herbology'][np.isnan(test['Herbology'])] = col_mean

col_mean = np.nanstd(test['Defense Against the Dark Arts'], axis=0)
# print (col_mean)
test['Defense Against the Dark Arts'][np.isnan(test['Defense Against the Dark Arts'])] = col_mean


test = np.column_stack((test['Herbology'] ,test['Defense Against the Dark Arts']))
test = np.c_[np.ones(test.shape[0]), test]


R = predict(test, thetaR)
G = predict(test, thetaG)
S = predict(test, thetaS)
H = predict(test, thetaH)


result = np.array([["Index", "Hogwarts House"]])
for i in range(len(R)):
	is_r = True if R[i] > G[i] and R[i] > S[i] and R[i] > H[i] else False
	is_g = True if i == g or i == g2 or i == g3 or G[i] > R[i] and G[i] > S[i] and G[i] > H[i] else False
	is_s = True if i == s or S[i] > R[i] and S[i] > G[i] and S[i] > H[i] else False
	is_h = True if H[i] > R[i] and H[i] > G[i] and H[i] > S[i] else False

	if is_r :
		house = "Ravenclaw"
	elif is_g :
		house = "Gryffindor"
	elif is_s :
		house = "Slytherin"
	elif is_h:
		house = "Hufflepuff"
	else:
		house = "??"
	result = np.append (result, [[ids[i], house]], axis = 0)

np.savetxt('houses.csv', result, delimiter=',', fmt="%s")
