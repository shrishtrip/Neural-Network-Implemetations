# importing libraries
import pandas as pd
import numpy as np
import random
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, AveragePooling1D
from sklearn.model_selection import train_test_split


# input from file
df1 = pd.read_csv('ecg_in_window.csv', delim_whitespace=False, sep=',', header=None)
df1 = df1.values.tolist()
df2 = pd.read_csv('label.csv', delim_whitespace=False, sep=',', header=None)
df2 = df2.values.tolist()

# shuffle
random.Random(4).shuffle(df1)
random.Random(4).shuffle(df2)

# divide in test and train
train, test = train_test_split(df1, test_size=0.2)
train_Y, test_Y = (train_test_split(df2, test_size=0.2))


# Normalize the data set
scaler = MinMaxScaler().fit(train)
train = (scaler.transform(train))
scaler = MinMaxScaler().fit(test)
test = (scaler.transform(test))
train  = np.expand_dims(train, axis=2)
test  = np.expand_dims(test, axis=2)

train_Y = np.array(train_Y)
test_Y = np.array(test_Y)

# Converts a list into a matrix with p columns, one for each category
train_Y = keras.utils.to_categorical(train_Y, num_classes=2)
test_Y = keras.utils.to_categorical(test_Y, num_classes=2)

# Define the learning rate
Learning_Rate = 1.0e-8


model = Sequential()
# Convolution Layer #1
model.add(Convolution1D(100,6,strides=1,padding='same', data_format='channels_last',activation='sigmoid',use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',input_shape=(1000,1)))         # , input_shape=(28,28,1)
# Pooling Layer #1
model.add(AveragePooling1D(pool_size=2, strides=2, padding='same', data_format='channels_last'))
model.add(Activation('tanh'))
# Convolution Layer #2
model.add(Convolution1D(50,6,strides=1,padding='same', data_format='channels_last',activation='sigmoid',use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', input_shape=(500,1,100)))        # , input_shape=(28,28,1)
# Pooling Layer #2
model.add(AveragePooling1D(pool_size=2, strides=2, padding='same', data_format='channels_last'))
model.add(Activation('tanh'))
model.add(Flatten())
# Fully Connected Layer #1
model.add(Dense(50))
model.add(Dropout(0.2))
# Fully Connected Layer #2
model.add(Dense(25))
model.add(Dropout(0.2))
# Fully Connected Layer #3
model.add(Dense(10))
model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation('sigmoid'))
model.compile(loss='mean_squared_error', optimizer = 'adadelta', metrics = ['accuracy'])

print(model.summary())
model_details = model.fit(train,train_Y,epochs=1)  #,validation_data=(test,test_Y)
print(model.evaluate(test,test_Y))






