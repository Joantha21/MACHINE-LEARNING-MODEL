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

data = pd.read_csv('emotions.csv')
data.describe()

#Separarting Positive, Neagtive and Neutral dataframes:
pos = data.loc[data["label"] == "POSITIVE"]
sample_pos = pos.loc[2, 'fft_0_b':'fft_749_b']
neg = data.loc[data["label"] == "NEGATIVE"]
sample_neg = neg.loc[0, 'fft_0_b':'fft_749_b']
neu = data.loc[data["label"] == "NEUTRAL"]
sample_neu = neu.loc[1, 'fft_0_b':'fft_749_b']

def Transform_data(data):
    #Encoding Lables into numbers
    encoding_data = ({'NEUTRAL': 0, 'POSITIVE': 1, 'NEGATIVE': 2} )
    data_encoded = data.replace(encoding_data)
    #getting brain signals into x variable
    x = data_encoded.drop(["label"]  ,axis=1)
    #getting labels into y variable
    y = data_encoded.loc[:,'label'].values
    scaler = StandardScaler()
    #scaling Brain Signals
    scaler.fit(x)
    X = scaler.transform(x)
    #One hot encoding Labels 
    Y = to_categorical(y)
    return X,Y

#Create pie chart for distribution
counts = data['label'].value_counts()
dlabels = {'NEUTRAL': 0, 'POSITIVE': 1, 'NEGATIVE': 2}
dlabels = [dlabels[label] for label in counts.index]

plt.figure(figsize = (8,8))
plt.pie(counts, labels = dlabels, autopct = '%1.1f%%',startangle = 120, colors = ['red','blue','green'])
plt.title('Pie Chart')
plt.axis('equal')
plt.show()

#spectral analysis
sampling_rates = 256
start = data.columns.get_loc('fft_0_b')
end   = data.columns.get_loc('fft_749_b') + 1

sample = data.iloc[0, start:end].to_numpy(dtype=float)

frequency, p_density = signal.welch(sample, fs=float(sampling_rates))
plt.figure(figsize=(10,6))
plt.figure(figsize=(10,6))
plt.semilogy(frequency,p_density)
plt.title('Gaylord andrei')
plt.grid(True)
plt.show()

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
# history = DaddyChill.fit(x_train, y_train, epochs = 10, validation_split = 0.1)
# loss, acc = DaddyChill.evaluate(x_test, y_test)