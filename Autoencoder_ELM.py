import math
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy.special import expit
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn.preprocessing import MinMaxScaler



def activation(b, w, x):
    sum = b;
    for j in range(len(x)):
        sum+= w[j] * x[j]
    return calc(sum);

def calc(u):
    t = 1 + math.exp(-u)
    return 1/t

def ELM(xtrain,ytrain,xtest,ytest):
    
    ##TRAINING
    Y=[]
    for i in ytrain:
        if i==0:
            Y.append([1,0])
        else:
            Y.append([0,1])
            
    ytrain = Y
    c=8    #no_of_hidden_neurons
    m=len(xtrain)   #no_of_training_data
    n=len(xtrain[0])    #no_of_input_neurons
    
    randommat = np.random.randn(n+1,c)
    temp = np.concatenate((np.ones((len(xtrain),1)), xtrain), axis =1)
    tempH = np.dot(temp, randommat)
    H = np.cos(tempH)
    w= np.linalg.pinv(H)
    W = np.dot(w, ytrain)

     ##TESTING
     
    Y=[]
    for i in ytest:
        if i==0:
            Y.append([1,0])
        else:
            Y.append([0,1])
    ytest= Y
    temp = np.concatenate((np.ones((len(xtest),1)), xtest), axis =1)
    testH = np.dot(temp, randommat)
    Ht = np.cos(testH)
    yn = np.dot(Ht, W)
    lp = []
    lt = []
    for i in range(len(xtest)):
        if(yn[i][0]>yn[i][1]):
            lp.append(0)
        else:
            lp.append(1)
        if(ytest[i][0]>ytest[i][1]):
            lt.append(0)
        else:
            lt.append(1)
    cmt = confusion_matrix(lp, lt)
    summ = sum(sum(cmt[:]))
    ac = (cmt[0][0] + cmt[1][1])/summ
    return ac

#reading data
df = pd.read_excel('data5.xlsx', header=None).values

# divide in test and train
random.shuffle(df)
train, test = train_test_split(df, test_size=0.2)
train = np.asarray(train)
test = np.array(test)

ytest=test[:,-1]    
xtest=test[:,:-1]
ytrain =train[:,-1]    
xtrain=train[:,:-1]


Y = ytrain
train = xtrain

Y_test = ytest
test = xtest

#Scaling
scaler = MinMaxScaler().fit(train)
train = scaler.transform(train)

scaler = MinMaxScaler().fit(test)
test = scaler.transform(test)

Y = Y.tolist()
train = train.tolist()
Y_test = Y_test.tolist()
test = test.tolist()

no_of_iterations = 10
eta = -0.00001




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

    def plot_cost(self):
        plt.plot(self.X_axis, self.Y_axis)
        plt.show()

    def Reduced_input(self,input_mat):
        ans = []
        for i in input_mat:
            X = i
            U = np.dot(self.weight1, X)
            U = U + self.bias1
            H = expit(U)
            # V = np.dot(self.weight2, H)
            # V = V + self.bias2
            # Y_pred = expit(V)
            ans.append(H)
        return  list(ans)





    def iterate(self):
        self.weight2 = (np.array(self.weight1).transpose()).tolist()
        # self.weight2 = self.weight1.transpose()
        for itr in range(no_of_iterations):
            print("iteration", itr)
            self.X_axis.append(itr)
            MSE = 0
            for i in range(len(self.input_matrix)):
                # Forward
                X = self.input_matrix[i]
                U = np.dot(self.weight1, X)
                U = U + self.bias1
                H = expit(U)
                V = np.dot(self.weight2, H)
                V = V + self.bias2
                Y_pred = expit(V)

                # BACKWARD #
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


##########Autoencoders Training
AE1 = Autoencoder(72,25,train)
AE1.initialise_weights()
AE1.iterate()

temp = AE1.Reduced_input(train)
AE2 = Autoencoder(25,10,temp)
AE2.initialise_weights()
AE2.iterate()

#reducing dimensionality for test data
AE1_test = Autoencoder(72,25,test)
AE1_test.initialise_weights()
AE1_test.iterate()

temp2 = AE1_test.Reduced_input(test)
AE2_test = Autoencoder(25,10,temp)
AE2_test.initialise_weights()
AE2_test.iterate()



test = AE2_test.Reduced_input(temp2)
Y_pred = AE2.Reduced_input(temp)

#ELM Layer
acc=ELM(Y_pred,Y,test,Y_test)

print("Accuracy- ",acc)
