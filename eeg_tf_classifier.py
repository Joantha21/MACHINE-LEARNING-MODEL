import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import layers, Model
import mne
from mne.time_frequency import psd_array_welch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,classification_report,top_k_accuracy_score,accuracy_score


data = pd.read_csv("eeg.csv")  


ENCODING = {"delta":0, "theta":1, "alpha":2, "beta":3, "gamma":4}
CLASS_NAMES = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
BANDS_HZ = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 50.0),
}

def compute_dominant_hz(signal, sfreq, fmin=1.0, fmax=50.0):
    """
    Return the dominant (peak) frequency in Hz for a 1D signal using Welch PSD (MNE).
    signal: 1D array-like (samples)
    sfreq: sampling frequency (Hz)
    """
    sig = np.asarray(signal, dtype=np.float32)
    if sig.ndim != 1:
        raise ValueError("signal must be 1D (n_samples,)")

    # PSD via MNE (Welch). Shape: (1, n_freqs)
    psd, freqs = psd_array_welch(
        sig[np.newaxis, :],
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        average="mean",
        n_fft=None,       # let MNE choose
        n_overlap=0,
        window="hamming",
        verbose=False
    )
    psd = psd[0]  # (n_freqs,)
    peak_idx = int(np.argmax(psd))
    return float(freqs[peak_idx])

def compute_bandpowers_mne(signal, sfreq):
    """
    Compute approximate band powers for Delta/Theta/Alpha/Beta/Gamma using MNE Welch PSD.
    Returns a dict with keys: ['Delta','Theta','Alpha','Beta','Gamma'].
    """
    sig = np.asarray(signal, dtype=np.float32)
    psd, freqs = psd_array_welch(
        sig[np.newaxis, :],
        sfreq=sfreq,
        fmin=1.0,
        fmax=50.0,
        average="mean",
        n_fft=None,
        n_overlap=0,
        window="hamming",
        verbose=False
    )
    psd = psd[0]  # (n_freqs,)
    # Frequency resolution (approx) for integrating PSD
    if len(freqs) > 1:
        df = float(np.diff(freqs).mean())
    else:
        df = 1.0

    def band_power(low, high):
        mask = (freqs >= low) & (freqs < high)
        # approximate area under PSD in band
        return float(np.sum(psd[mask]) * df)

    return {
        "Delta": band_power(*BANDS_HZ["delta"]),
        "Theta": band_power(*BANDS_HZ["theta"]),
        "Alpha": band_power(*BANDS_HZ["alpha"]),
        "Beta":  band_power(*BANDS_HZ["beta"]),
        "Gamma": band_power(*BANDS_HZ["gamma"]),
    }

def predict_waveform_by_hz(hz):
    """
    Map a frequency (Hz) to the canonical band name with Hz label.
    """
    h = float(hz)
    if 1.0 <= h < 4.0:   return "Delta"
    if 4.0 <= h < 8.0:   return "Theta"
    if 8.0 <= h < 13.0:  return "Alpha"
    if 13.0 <= h < 30.0: return "Beta"
    if 30.0 <= h <= 50.0:return "Gamma"
    return f"Out of range ({h:.2f} Hz)"

def Transform_data(df: pd.DataFrame):
    
    # Normalize column access (case-insensitive)
    cols = {c.lower(): c for c in df.columns}
    required = ["delta", "theta", "alpha", "beta", "gamma"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

    band_cols = [cols[c] for c in required]

    # Features
    X_raw = df[band_cols].to_numpy(dtype=np.float32)

    # Labels via argmax across the 5 bands
    y_idx = X_raw.argmax(axis=1).astype("int32")  # 0..4

    # One-hot for categorical_crossentropy
    Y = tf.keras.utils.to_categorical(y_idx, num_classes=5)

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    return X, Y

# -------------------- MODEL --------------------
def create_model(input_dim, num_classes, model_name="Daddy"):
    inputs = layers.Input(shape=(input_dim,), name="inputs")
    # Treat the 5 features as a tiny sequence of length=5 with feature_dim=1
    x = layers.Lambda(lambda t: tf.expand_dims(t, axis=-1), name="expand_dims")(inputs)
    x = layers.GRU(256, return_sequences=True, name="gru")(x)
    x = layers.Flatten(name="flatten")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="dense")(x)
    return Model(inputs, outputs, name=model_name)

try:
    pd.set_option('future.no_silent_downcasting', True)
except Exception:
    pass  
# Ensure `data` exists; if not, fail clearly
if 'data' not in globals():
    raise RuntimeError("You must provide a DataFrame named `data` with columns Delta, Theta, Alpha, Beta, Gamma.")

X, Y = Transform_data(data)
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=4, stratify=Y.argmax(axis=1)
)

# Ensure numpy arrays
x_train = np.asarray(x_train); x_test = np.asarray(x_test)
y_train = np.asarray(y_train); y_test = np.asarray(y_test)

# Inspect class imbalance
y_train_idx = y_train.argmax(axis=1)
y_test_idx  = y_test.argmax(axis=1)

counts_train = np.bincount(y_train_idx, minlength=5)
counts_test  = np.bincount(y_test_idx,  minlength=5)
print("Train class counts:", counts_train)   # [Delta,Theta,Alpha,Beta,Gamma]
print("Test  class counts:", counts_test)

# Inverse-frequency sample weights (balanced)
class_weights = counts_train.sum() / (len(counts_train) * np.maximum(counts_train, 1))
sample_weight = class_weights[y_train_idx].astype("float32")

num_classes = y_train.shape[1]
assert num_classes == 5, f"Expected 5 classes; got {num_classes}"

DaddyChill = create_model(x_train.shape[1], num_classes, model_name="Daddy")
DaddyChill.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
DaddyChill.summary()

history = DaddyChill.fit(x_train, y_train, epochs=40, validation_split=0.1, verbose=1)
loss, acc = DaddyChill.evaluate(x_test, y_test, verbose=0)
print(f"Test loss={loss:.4f}  acc={acc:.4f}")

pred_idx = DaddyChill.predict(x_test[:5], verbose=0).argmax(axis=1)
print("Model predictions (first 5):", [CLASS_NAMES[i] for i in pred_idx])

# ---- 1) Get predictions & basics ----
y_true = y_test.argmax(axis=1)
proba  = DaddyChill.predict(x_test, verbose=0)
y_pred = proba.argmax(axis=1)

num_classes = proba.shape[1]
labels_idx = np.arange(num_classes)

top1 = accuracy_score(y_true, y_pred)
top2 = top_k_accuracy_score(y_true, proba, k=2, labels=labels_idx)

print(f"Top-1 acc: {top1:.4f}")
print(f"Top-2 acc: {top2:.4f}")
print("\nClassification report (zero_division=0 to avoid warnings):\n")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

# ---- 2) Confusion matrix (counts) ----
cm = confusion_matrix(y_true, y_pred, labels=labels_idx)

plt.figure(figsize=(6,5))
plt.imshow(cm)
plt.title("Confusion Matrix (counts)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(labels_idx, CLASS_NAMES, rotation=45, ha="right")
plt.yticks(labels_idx, CLASS_NAMES)
# annotate counts
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.tight_layout()
plt.show()

# ---- 3) Confusion matrix (row-normalized) ----
with np.errstate(invalid="ignore", divide="ignore"):
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums>0)

plt.figure(figsize=(6,5))
plt.imshow(cm_norm)
plt.title("Confusion Matrix (row-normalized)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(labels_idx, CLASS_NAMES, rotation=45, ha="right")
plt.yticks(labels_idx, CLASS_NAMES)

# annotate with percentages
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, f"{cm_norm[i, j]*100:.0f}%", ha="center", va="center")
plt.tight_layout()
plt.show()

# ---- 4) Per-class Precision / Recall / F1 (three separate figures) ----
prec, rec, f1, sup = precision_recall_fscore_support(
    y_true, y_pred, labels=labels_idx, zero_division=0
)

# Precision
plt.figure(figsize=(7,4))
plt.bar(np.arange(num_classes), prec)
plt.title("Per-class Precision")
plt.xticks(labels_idx, CLASS_NAMES, rotation=45, ha="right")
plt.ylabel("Precision")
plt.ylim(0, 1.0)
plt.tight_layout()
plt.show()

# Recall
plt.figure(figsize=(7,4))
plt.bar(np.arange(num_classes), rec)
plt.title("Per-class Recall")
plt.xticks(labels_idx, CLASS_NAMES, rotation=45, ha="right")
plt.ylabel("Recall")
plt.ylim(0, 1.0)
plt.tight_layout()
plt.show()

# F1
plt.figure(figsize=(7,4))
plt.bar(np.arange(num_classes), f1)
plt.title("Per-class F1-score")
plt.xticks(labels_idx, CLASS_NAMES, rotation=45, ha="right")
plt.ylabel("F1-score")
plt.ylim(0, 1.0)
plt.tight_layout()
plt.show()

# ---- 5) Confidence histogram (max softmax) for correct vs incorrect ----
max_conf = proba.max(axis=1)
is_correct = (y_pred == y_true)

plt.figure(figsize=(7,4))
plt.hist(max_conf[is_correct], bins=20, alpha=0.7, label="Correct")
plt.hist(max_conf[~is_correct], bins=20, alpha=0.7, label="Incorrect")
plt.title("Prediction Confidence (Max Softmax)")
plt.xlabel("Confidence")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.show()

# compute for each sample if the true class is in top-2 predicted
top2_pred_idx = np.argsort(-proba, axis=1)[:, :2]
top2_hit = np.array([y_true[i] in top2_pred_idx[i] for i in range(len(y_true))], dtype=bool)

per_class_top1 = []
per_class_top2 = []
for c in labels_idx:
    mask = (y_true == c)
    if mask.sum() == 0:
        per_class_top1.append(0.0)
        per_class_top2.append(0.0)
    else:
        per_class_top1.append((y_pred[mask] == c).mean())
        per_class_top2.append(top2_hit[mask].mean())

plt.figure(figsize=(8,4))
x = np.arange(num_classes)
width = 0.35
plt.bar(x - width/2, per_class_top1, width, label="Top-1")
plt.bar(x + width/2, per_class_top2, width, label="Top-2")
plt.title("Per-class Top-1 vs Top-2 Accuracy")
plt.xticks(labels_idx, CLASS_NAMES, rotation=45, ha="right")
plt.ylim(0, 1.0)
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()