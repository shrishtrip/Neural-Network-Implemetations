import pandas as pd
from scipy.special import expit
from numpy import linalg as LA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import random


# input from file
df = pd.read_csv('data.csv', delim_whitespace=False, sep=',', header=None)
df = df.values.tolist()

# divide in test and train
train, test = train_test_split(df, test_size=0.2)
train = np.asarray(train)
test = np.array(test)

Y = train[:,72]
train = train[:,:-1]

Y_test = test[:,72]
test = test[:,:-1]

# Normalise the data set
scaler = MinMaxScaler().fit(train)
train = scaler.transform(train)
scaler = MinMaxScaler().fit(test)
test = scaler.transform(test)

Y = Y.tolist()
train = train.tolist()
Y_test = Y_test.tolist()
test = test.tolist()

no_of_iterations = 1000
eta = -0.00001



# Class Aautoencoder with weights bias init(),iterate(), plot_cost() and reduced_input() methods
class Autoencoder():

    def __init__(self,n1,n2,ls):
        self.input_matrix = []
        self.no_of_input_neurons = 0
        self.no_of_hidden_neurons = 0
        self.X_axis = []
        self.Y_axis = []
        self.weight1 = []
        self.weight2 = []
        self.bias1 = []
        self.bias2 = []
        self.no_of_input_neurons = n1
        self.no_of_hidden_neurons = n2
        self.input_matrix = ls

    # initialise the weights
    def initialise_weights(self):
        for i in range(self.no_of_hidden_neurons):
            ls = []
            ls.clear()
            limit = np.sqrt(self.no_of_input_neurons)
            limit = 1 / limit
            for j in range(self.no_of_input_neurons):
                ls.append(round(random.uniform(-1 * limit, limit), 3))
            self.weight1.append(ls)

        for i in range(self.no_of_hidden_neurons):
            self.bias1.append(round(random.uniform(-1 * limit, limit), 3))

        for i in range(self.no_of_input_neurons):
            self.bias2.append(round(random.uniform(-1 * limit, limit), 3))

    # Plot the cost vs iterations
    def plot_cost(self):
        plt.plot(self.X_axis, self.Y_axis)
        plt.show()

    # Return the output from the hidden layer
    def Reduced_input(self,input_mat):
        ans = []
        for i in input_mat:
            X = i
            U = np.dot(self.weight1, X)
            U = U + self.bias1
            H = expit(U)
            ans.append(H)
        return  list(ans)

    # Iterate to update the weights
    def iterate(self):
        self.weight2 = (np.array(self.weight1).transpose()).tolist()
        for itr in range(no_of_iterations):
            print("iteration", itr)
            self.X_axis.append(itr)
            MSE = 0
            for i in range(len(self.input_matrix)):

                # Forward Pass
                X = self.input_matrix[i]
                U = np.dot(self.weight1, X)
                U = U + self.bias1
                H = expit(U)
                V = np.dot(self.weight2, H)
                V = V + self.bias2
                Y_pred = expit(V)

                # BACKWARD Pass
                e = X - Y_pred
                MSE += LA.norm(e)**2
                E = e * (Y_pred * (1 - Y_pred))

                J_b02 = E
                colu = np.reshape(np.array(E), (np.array(E).shape[0], 1))
                rowv = np.reshape(np.array(H), (1, np.array(H).shape[0]))
                res = np.dot(colu, rowv)
                J_w02 = res
                self.bias2 -= eta * (J_b02)
                self.weight2 -= eta * (J_w02)

                temp = H * (1 - H)
                # temp = temp * E
                res = np.dot(self.weight1,E)
                J_b01 = res*temp
                self.bias1 -= eta * (J_b01)
                colu = np.reshape(np.array(J_b01), (np.array(J_b01).shape[0], 1))
                rowv = np.reshape(np.array(X), (1, np.array(X).shape[0]))
                res = np.dot(colu, rowv)
                J_w01 = res
                self.weight1 -= eta*(J_w01)
            MSE /= len(self.input_matrix)
            self.Y_axis.append(MSE)


# Instanciating  the Autoencoder
AE1 = Autoencoder(72,25,train)
# Initialise the weights
AE1.initialise_weights()
# Iterate for updations of weights
AE1.iterate()
# Plot the graphs
AE1.plot_cost()

temp = AE1.Reduced_input(train)
AE2 = Autoencoder(25,10,temp)
AE2.initialise_weights()
AE2.iterate()
AE2.plot_cost()

temp2 = AE2.Reduced_input(temp)
AE3 = Autoencoder(10,1,temp2)
AE3.initialise_weights()
AE3.iterate()
AE3.plot_cost()

Y_pred = AE3.Reduced_input(temp2)

for i in range(len(Y_pred)):
    if Y_pred[i]>=float(0.5):
        Y_pred[i] = 1
    else:
        Y_pred[i] = 0

# Testing the Model
ans = 0
for i in range(len(Y_test)):
    if Y_test[i]==Y_pred[i]:
       ans += 1


print(len(Y_test))
print(ans)
print("Accuracy",ans/len(Y_test))





























