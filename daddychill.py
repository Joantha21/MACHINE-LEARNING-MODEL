import numpy as np # linear algebra
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow

from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python import keras
from keras.utils import to_categorical
from keras import layers, Model
from keras import Sequential
from keras.optimizers import SGD

from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
tf.keras.backend.clear_session()
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import datasets, tree, linear_model, svm
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

import seaborn as sns
import time
import math
import pickle

data = pd.read_csv('eeg.csv')
data.describe()

#Calling above function and splitting dataset into train and test
pd.set_option('future.no_silent_downcasting', True)
X,Y = Transform_data(data)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)

x_train.shape[1]
len(x_train)

x_train = np.asarray(x_train); x_test = np.asarray(x_test)
y_train = np.asarray(y_train); y_test = np.asarray(y_test)

num_classes = int(np.unique(y_train).size)
assert y_train.shape[1] == 3
num_classes = 3

def create_model(input_dim, num_classes, model_name = "Daddy"):
    inputs = layers.Input(shape = (input_dim,), name = "inputs")
    x = layers.Lambda(lambda t: tf.expand_dims(t, axis = -1), name = "expand_dims")(inputs)  
    x = layers.GRU(256, return_sequences = True, name = "gru")(x)  
    x = layers.Flatten(name = "flatten")(x)                      
    outputs = layers.Dense(num_classes, activation = "softmax", name = "dense")(x)
    
    return Model(inputs, outputs, name = model_name)

DaddyChill = create_model(x_train.shape[1], num_classes)
DaddyChill.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
DaddyChill.summary()

#Training and Evaluting model
history = DaddyChill.fit(x_train, y_train, epochs = 10, validation_split = 0.1)
loss, acc = DaddyChill.evaluate(x_test, y_test)