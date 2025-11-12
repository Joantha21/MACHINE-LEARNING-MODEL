import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.python import keras
from keras import Model, layers

# Load data
data = pd.read_csv('eeg1.csv')

# Check data shape and info
print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
print(f"Shape: {data.shape}")
print(f"\nColumns: {list(data.columns)}")
print(f"\nFirst few rows:")
print(data.head())
print(f"\nBasic statistics:")
print(data.describe())

# Visualize sample EEG bands
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Plot first sample
sample = data.iloc[0]
axes[0].plot(sample.values, marker='o', linewidth=2, markersize=8)
axes[0].set_xticks(range(5))
axes[0].set_xticklabels(['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'])
axes[0].set_title("Sample EEG Band Powers (First Row)", fontsize=14, fontweight='bold')
axes[0].set_ylabel("Power", fontsize=12)
axes[0].grid(True, alpha=0.3)

# Plot average across all samples
avg_bands = data.mean()
axes[1].bar(range(5), avg_bands.values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
axes[1].set_xticks(range(5))
axes[1].set_xticklabels(['Delta\n(1-4Hz)', 'Theta\n(4-8Hz)', 'Alpha\n(8-13Hz)', 'Beta\n(13-30Hz)', 'Gamma\n(30-60Hz)'])
axes[1].set_title("Average EEG Band Powers Across All Samples", fontsize=14, fontweight='bold')
axes[1].set_ylabel("Average Power", fontsize=12)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Define labels and frequency bands
label_map = {0: 'delta', 1: 'theta', 2: 'alpha', 3: 'beta', 4: 'gamma'}
bands_hz = [(1,4), (4,8), (8,13), (13,30), (30,60)]

# Create label mapping for 3 classes (assuming you want to group the 5 bands into 3 classes)
label_mapping = {
    'low': 0,      # delta, theta
    'medium': 1,   # alpha
    'high': 2      # beta, gamma
}

def Transform_data(df):
    """
    Transform the EEG data into features (X) and labels (Y).
    Since the CSV has no label column, we'll create synthetic labels based on 
    dominant frequency bands.
    """
    # Extract features - all 5 EEG bands
    feature_cols = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    X = df[feature_cols].copy()
    
    # Create synthetic labels based on which frequency band is dominant
    # This is a reasonable approach for EEG data classification
    Y = df[feature_cols].idxmax(axis=1).map({
        'Delta': 0,   # Low frequency (1-4 Hz)
        'Theta': 0,   # Low frequency (4-8 Hz)
        'Alpha': 1,   # Medium frequency (8-13 Hz)
        'Beta': 2,    # High frequency (13-30 Hz)
        'Gamma': 2    # High frequency (30-60 Hz)
    })
    
    print(f"\nLabel distribution (based on dominant frequency):")
    print(Y.value_counts().sort_index())
    print(f"\nMapping: 0=Low (Delta/Theta), 1=Medium (Alpha), 2=High (Beta/Gamma)")
    
    return X, Y

# Transform data
pd.set_option('future.no_silent_downcasting', True)
X, Y = Transform_data(data)

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)

# Convert to numpy arrays
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

# Determine number of classes
num_classes = int(np.unique(y_train).size)
print(f"Number of classes: {num_classes}")
print(f"Training samples: {x_train.shape[0]}, Test samples: {x_test.shape[0]}")
print(f"Feature dimension: {x_train.shape[1]}")

def create_model(input_dim, num_classes, model_name="DaddyChill"):
    """Create an improved GRU-based model for EEG classification"""
    inputs = layers.Input(shape=(input_dim,), name="inputs")
    
    # Expand dimensions for RNN processing
    x = layers.Lambda(lambda t: tf.expand_dims(t, axis=-1), name="expand_dims")(inputs)  
    
    # First GRU layer with dropout
    x = layers.GRU(128, return_sequences=True, name="gru1")(x)
    x = layers.Dropout(0.3, name="dropout1")(x)
    
    # Second GRU layer
    x = layers.GRU(64, return_sequences=False, name="gru2")(x)
    x = layers.Dropout(0.3, name="dropout2")(x)
    
    # Dense layers
    x = layers.Dense(32, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2, name="dropout3")(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)
    
    return Model(inputs, outputs, name=model_name)

# Create model
DaddyChill = create_model(x_train.shape[1], num_classes)
DaddyChill.summary()

# Compile model
DaddyChill.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
print("\nTraining model...")
history = DaddyChill.fit(
    x_train,
    y_train,
    validation_split=0.2,
    batch_size=32,
    epochs=100,  # Increased epochs
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,  # Increased patience
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ],
    verbose=1
)

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[1].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Evaluate on test set
print("\nEvaluating on test set...")
probs = DaddyChill.predict(x_test, verbose=0)
y_pred = np.argmax(probs, axis=1)

acc = (y_pred == y_test).mean()
print(f"Test Accuracy: {acc * 100:.3f}%")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
clr = classification_report(y_test, y_pred, target_names=list(label_mapping.keys()))

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
plt.xticks(np.arange(num_classes) + 0.5, list(label_mapping.keys()), fontsize=11)
plt.yticks(np.arange(num_classes) + 0.5, list(label_mapping.keys()), rotation=0, fontsize=11)
plt.xlabel("Predicted Label", fontsize=12, fontweight='bold')
plt.ylabel("True Label", fontsize=12, fontweight='bold')
plt.title("Confusion Matrix - EEG Frequency Band Classification", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(clr)
print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"Test Accuracy: {acc * 100:.2f}%")
print(f"Number of test samples: {len(y_test)}")
print(f"Correctly classified: {(y_pred == y_test).sum()}")
print(f"Misclassified: {(y_pred != y_test).sum()}")
print("=" * 60)