
import argparse, json, math, os, random
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import tensorflow as tf

DEFAULT_LABEL_MAP = {0:"delta",1:"theta",2:"alpha",3:"beta",4:"gamma"}

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def synth_band_signal(batch:int, C:int, T:int, fs:int, band:Tuple[float,float]) -> np.ndarray:
    low, high = band; t = np.arange(T)/fs
    out = np.zeros((batch,C,T), dtype=np.float32)
    for b in range(batch):
        freqs = np.random.uniform(low, high, size=(C,2))
        phases= np.random.uniform(0, 2*np.pi, size=(C,2))
        amps  = np.random.uniform(0.7,1.3, size=(C,2))
        for c in range(C):
            sig = np.zeros_like(t, dtype=np.float32)
            for k in range(2):
                sig += (amps[c,k]*np.sin(2*np.pi*freqs[c,k]*t + phases[c,k])).astype(np.float32)
            sig += np.random.normal(0,0.1,size=T).astype(np.float32)
            out[b,c]=sig
    return out

def bandpowers_from_timeseries(X: np.ndarray, fs:int, bands:Dict[int,Tuple[float,float]]) -> np.ndarray:
    N,C,T = X.shape; freqs = np.fft.rfftfreq(T, d=1.0/fs)
    bp = np.zeros((N,C,5), dtype=np.float32)
    for i in range(N):
        Xf = np.abs(np.fft.rfft(X[i], axis=-1))**2 / T
        for bi,(lo,hi) in enumerate([bands[0],bands[1],bands[2],bands[3],bands[4]]):
            mask = (freqs >= lo) & (freqs < hi)
            bp[i,:,bi] = Xf[:,mask].mean(axis=1) if mask.any() else 0.0
    return bp

def make_demo_dataset(N=3000, C=4, T=512, fs=128):
    bands = {0:(0.5,4.0),1:(4.0,8.0),2:(8.0,12.0),3:(13.0,30.0),4:(30.0,45.0)}
    per = N//5; X_list=[]; y_list=[]
    for cls,band in bands.items():
        Xc = synth_band_signal(per, C, T, fs, band); X_list.append(Xc); y_list.append(np.full((per,),cls,np.int64))
    X = np.concatenate(X_list, axis=0); y = np.concatenate(y_list, axis=0)
    idx = np.random.permutation(X.shape[0]); X=X[idx]; y=y[idx]
    X_bp = bandpowers_from_timeseries(X, fs, bands)  # [N,C,5]
    return X_bp.astype(np.float32), y.astype(np.int64)

def compute_bp_stats(X_bp: np.ndarray):
    X_log = np.log1p(np.maximum(X_bp, 0.0))
    mu = X_log.mean(axis=0).astype(np.float32)  # [C,5]
    sd = X_log.std(axis=0).astype(np.float32); sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd

def standardize_bp(X_bp: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    X_log = np.log1p(np.maximum(X_bp, 0.0)).astype(np.float32)
    return (X_log - mu[None,...]) / sd[None,...]

def build_bp_model(C:int, num_classes:int):
    inp = tf.keras.Input(shape=(C,5), dtype=tf.float32)
    x = tf.keras.layers.Flatten()(inp)
    x = tf.keras.layers.Dense(128, activation=tf.nn.gelu)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation=tf.nn.gelu)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def main():
    ap = argparse.ArgumentParser(description="EEG band-power classifier (TensorFlow)")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--data", type=str, default=None, help="NPZ with X_bp:[N,C,5], y:[N]")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--test-split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint", type=str, default="bp_model_best_tf.keras")
    ap.add_argument("--labels", type=str, default="bp_label_map.json")
    ap.add_argument("--stats", type=str, default="bp_stats.json")
    ap.add_argument("--predict-npy", type=str, default=None, help="One [C,5] window")
    ap.add_argument("--threshold", type=float, default=0.6)
    args = ap.parse_args(); set_seed(args.seed)

    if args.predict_npy is not None:
        model = tf.keras.models.load_model(args.checkpoint, compile=False)
        with open(args.labels,"r",encoding="utf-8") as f: label_map = {int(k):v for k,v in json.load(f).items()}
        with open(args.stats,"r",encoding="utf-8") as f: stats = json.load(f); mu=np.asarray(stats["mu"],np.float32); sd=np.asarray(stats["sd"],np.float32)
        x = np.load(args.predict_npy).astype(np.float32)   # [C,5]
        x = standardize_bp(x[None,...], mu, sd)            # [1,C,5]
        probs = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs)); conf=float(probs[idx]); label=label_map.get(idx,str(idx))
        out = {"pred_index":idx,"pred_label":label,"confidence":round(conf,4),"is_known":bool(conf>=args.threshold),
               "threshold":args.threshold,"probs":{label_map.get(i,str(i)):float(p) for i,p in enumerate(probs)}}
        print(json.dumps(out, indent=2)); return

    if args.demo:
        X_bp, y = make_demo_dataset(N=3000, C=4, T=512, fs=128); label_map = DEFAULT_LABEL_MAP.copy()
        print(f"[demo] X_bp: {X_bp.shape}, y: {y.shape}")
    else:
        assert args.data is not None, "Provide --data NPZ or use --demo"
        npz = np.load(args.data); X_bp = npz["X_bp"].astype(np.float32); y = npz["y"].astype(np.int64)
        if "label_map_json" in npz.files:
            label_map = {int(k): v for k, v in json.loads(str(npz['label_map_json'].tolist())).items()}
        else:
            classes = sorted(list(set(int(c) for c in np.unique(y).tolist())))
            label_map = {i: DEFAULT_LABEL_MAP.get(i, str(i)) for i in classes}
        print(f"[data] X_bp: {X_bp.shape}, y: {y.shape}")

    N,C,_ = X_bp.shape
    val_n = int(math.floor(args.val_split*N)); test_n = int(math.floor(args.test_split*N))
    train_n = N - val_n - test_n; assert train_n>0 and val_n>0 and test_n>0
    idx = np.random.permutation(N); tr_idx = idx[:train_n]; va_idx = idx[train_n:train_n+val_n]; te_idx = idx[train_n+val_n:]
    Xtr, ytr = X_bp[tr_idx], y[tr_idx]; Xva, yva = X_bp[va_idx], y[va_idx]; Xte, yte = X_bp[te_idx], y[te_idx]

    mu, sd = compute_bp_stats(Xtr)
    Xtr = standardize_bp(Xtr, mu, sd); Xva = standardize_bp(Xva, mu, sd); Xte = standardize_bp(Xte, mu, sd)

    tr_ds = tf.data.Dataset.from_tensor_slices((Xtr,ytr)).shuffle(min(10000,len(Xtr)), seed=args.seed).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    va_ds = tf.data.Dataset.from_tensor_slices((Xva,yva)).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    te_ds = tf.data.Dataset.from_tensor_slices((Xte,yte)).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    model = build_bp_model(C=C, num_classes=len(set(y.tolist())))
    ckpt = tf.keras.callbacks.ModelCheckpoint(args.checkpoint, monitor="val_accuracy", save_best_only=True, save_weights_only=False)
    es   = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
    rl   = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)
    model.fit(tr_ds, validation_data=va_ds, epochs=args.epochs, callbacks=[ckpt, es, rl], verbose=2)

    with open(args.labels,"w",encoding="utf-8") as f: json.dump({int(k):v for k,v in label_map.items()}, f, indent=2)
    with open(args.stats,"w",encoding="utf-8") as f: json.dump({"mu":mu.tolist(),"sd":sd.tolist()}, f, indent=2)
    print(f"Saved: {args.checkpoint}, {args.labels}, {args.stats}")

    best = tf.keras.models.load_model(args.checkpoint, compile=False)
    y_true, y_pred = [], []
    for xb,yb in te_ds:
        probs = best.predict(xb, verbose=0)
        y_true += yb.numpy().tolist(); y_pred += probs.argmax(axis=1).tolist()
    acc = accuracy_score(y_true, y_pred); f1 = f1_score(y_true, y_pred, average="macro")
    print(f"[Test] acc {acc:.3f} f1 {f1:.3f}")
    print("Confusion matrix (rows=true, cols=pred):"); print(confusion_matrix(y_true, y_pred))
    target_names = [label_map.get(i, str(i)) for i in sorted(set(y_true))]
    print(classification_report(y_true, y_pred, target_names=target_names, digits=3))

if __name__ == "__main__":
    main()
