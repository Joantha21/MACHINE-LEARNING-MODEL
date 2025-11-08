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
history = DaddyChill.fit(x_train, y_train, epochs = 10, validation_split = 0.1)
loss, acc = DaddyChill.evaluate(x_test, y_test)

# ================================
# EEG 5-band waveform classifier
# ================================
# We build band-power features (delta/theta/alpha/beta/gamma) from each row,
# create weak labels (argmax band), train a small MLP, and add an inference helper.

label_map = {0: 'delta', 1: 'theta', 2: 'alpha', 3: 'beta', 4: 'gamma'}
bands_hz = [(1,4), (4,8), (8,13), (13,30), (30,50)]
fs = int(sampling_rates)  # you set sampling_rates = 256 above

# Grab the per-row time series you already use ('fft_0_b'..'fft_749_b')
X_rows = data.loc[:, 'fft_0_b':'fft_749_b'].to_numpy(dtype=float)  # shape [N, T]


def row_bandpowers_welch(x_row, fs=fs):
    # Welch PSD from one 1D time series row
    f, Pxx = signal.welch(
        x_row.astype(float),
        fs=fs,
        nperseg=2*fs,
        noverlap=fs,
        nfft=2*fs,
        window='hann',
        detrend='constant',
        scaling='density',
        average='mean'
    )
    bp = []
    for (lo, hi) in bands_hz:
        mask = (f >= lo) & (f < hi)
        bp.append(Pxx[mask].mean() if mask.any() else 0.0)
    return np.array(bp, dtype=np.float32)  # [5]

# Build band-power matrix X_bp:[N,5]
X_bp = np.vstack([row_bandpowers_welch(row, fs=fs) for row in X_rows]).astype(np.float32)

# Log-scale + standardize band-powers
bp_scaler = StandardScaler().fit(np.log1p(X_bp))
X_bp_std = bp_scaler.transform(np.log1p(X_bp)).astype(np.float32)

# ---- WEAK labels (argmax band). Replace with your true labels if you have them. ----
y_bp = X_bp.argmax(axis=1).astype(np.int64)
Y_bp = to_categorical(y_bp, num_classes=5)

# Split
Xtr_bp, Xte_bp, Ytr_bp, Yte_bp = train_test_split(
    X_bp_std, Y_bp, test_size=0.2, random_state=4, stratify=y_bp
)

# Simple MLP over band-powers
def build_bp_model(input_dim, num_classes=5, name='WaveformBP'):
    inp = layers.Input(shape=(input_dim,), name='bp_in')
    x = layers.Dense(128, activation='gelu')(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='gelu')(x)
    x = layers.Dropout(0.1)(x)
    out = layers.Dense(num_classes, activation='softmax', name='bp_out')(x)
    return Model(inp, out, name=name)

WaveformBP = build_bp_model(Xtr_bp.shape[1], 5)
WaveformBP.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
WaveformBP.summary()

hist_bp = WaveformBP.fit(
    Xtr_bp, Ytr_bp,
    epochs=12, batch_size=128,
    validation_split=0.15, verbose=2
)

loss_bp, acc_bp = WaveformBP.evaluate(Xte_bp, Yte_bp, verbose=0)
print(f"[WaveformBP] Test accuracy: {acc_bp:.3f}")

# Report + confusion matrix
y_true_bp = np.argmax(Yte_bp, axis=1)
y_pred_bp = np.argmax(WaveformBP.predict(Xte_bp, verbose=0), axis=1)
print(classification_report(y_true_bp, y_pred_bp,
      target_names=[label_map[i] for i in range(4)], digits=3))
ConfusionMatrixDisplay.from_predictions(
    y_true_bp, y_pred_bp,
    display_labels=[label_map[i] for i in range(4)],
    xticks_rotation=45
)
plt.title("Waveform band classifier (weak labels)")
plt.show()

# ---------------- Inference helper ----------------
def predict_waveform_from_timeseries(ts_1d, fs=fs, threshold=0.6):
    """
    ts_1d: 1D NumPy array (one EEG window for one channel).
    Returns dict with label, confidence, is_known, and per-class probs.
    """
    f, Pxx = signal.welch(
        np.asarray(ts_1d, dtype=float),
        fs=fs, nperseg=2*fs, noverlap=fs, nfft=2*fs, window='hann',
        detrend='constant', scaling='density', average='mean'
    )
    bps = []
    for (lo, hi) in bands_hz:
        mask = (f >= lo) & (f < hi)
        bps.append(Pxx[mask].mean() if mask.any() else 0.0)
    bps = np.array(bps, dtype=np.float32)[None, :]  # [1,5]
    bps_std = bp_scaler.transform(np.log1p(bps))
    probs = WaveformBP.predict(bps_std, verbose=0)[0]
    idx = int(np.argmax(probs)); conf = float(probs[idx])
    return {
        "pred_index": idx,
        "pred_label": label_map[idx],
        "confidence": round(conf, 4),
        "is_known": bool(conf >= threshold),
        "threshold": threshold,
        "probs": {label_map[i]: float(p) for i, p in enumerate(probs)}
    }

# Demo prediction on your earlier 'sample' row:
demo_pred = predict_waveform_from_timeseries(sample, fs=fs, threshold=0.6)
print("Demo waveform prediction:", demo_pred)
