import math
import random
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def activation(b, w, x):
    sum = b;
    for j in range(len(x)):
        sum+= w[j] * x[j]
    return calc(sum);

def calc(u):
    t = 1 + math.exp(-u)
    return 1/t

def ELM(xtrain,ytrain,xtest,ytest,acc,sen,spe):
    
    ##TRAINING
    Y=[]
    for i in ytrain:
        if i==0:
            Y.append([1,0])
        else:
            Y.append([0,1])
            
    ytrain = Y
    c=30    #no_of_hidden_neurons
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
    se = (cmt[0][0]/(cmt[0][0]+cmt[0][1]))
    sp = (cmt[1][1]/(cmt[1][1]+cmt[1][0]))
    acc.append(ac)
    sen.append(se)
    spe.append(sp)
    

#Reading Data        
dataset=pd.read_csv('data.csv', header=None).values
random.shuffle(dataset)

#Implementing 5-Fold cross-validation technique
data1= dataset[:int(len(dataset)*0.2)]
data2= dataset[int(len(dataset)*0.2):int(len(dataset)*0.4)]
data3= dataset[int(len(dataset)*0.4):int(len(dataset)*0.6)]
data4= dataset[int(len(dataset)*0.6):int(len(dataset)*0.8)]
data5= dataset[int(len(dataset)*0.8):int(len(dataset))]


acc=[]
sen=[]
spe=[]

#splitting data into train and test

#using batch 1 as test
ytest=data1[:,-1]
xtest=data1[:,:-1]
data=np.concatenate((data2,data3,data4,data5), axis=0)
ytrain = data[:,-1]
xtrain=data[:,:-1]
ELM(xtrain,ytrain,xtest,ytest,acc,sen,spe)

#using batch 2 as test
ytest=data2[:,-1]
xtest=data2[:,:-1]
data=np.concatenate((data1,data3,data4,data5), axis=0)
ytrain = data[:,-1]
xtrain=data[:,:-1]
ELM(xtrain,ytrain,xtest,ytest,acc,sen,spe)

#using batch 3 as test
ytest=data3[:,-1]
xtest=data3[:,:-1]
data=np.concatenate((data2,data1,data4,data5), axis=0)
ytrain = data[:,-1]
xtrain=data[:,:-1]
ELM(xtrain,ytrain,xtest,ytest,acc,sen,spe)

#using batch 4 as test
ytest=data4[:,-1]
xtest=data4[:,:-1]
data=np.concatenate((data2,data3,data1,data5), axis=0)
ytrain = data[:,-1]
xtrain=data[:,:-1]
ELM(xtrain,ytrain,xtest,ytest,acc,sen,spe)

#using batch 5 as test
ytest=data5[:,-1]
xtest=data5[:,:-1]
data=np.concatenate((data2,data3,data4,data1), axis=0)
ytrain = data[:,-1]
xtrain=data[:,:-1]
ELM(xtrain,ytrain,xtest,ytest,acc,sen,spe)


#Computing average Accuracy, Specificity and Sensitivity
print(np.average(acc))
print(np.average(sen))
print(np.average(spe))
    

