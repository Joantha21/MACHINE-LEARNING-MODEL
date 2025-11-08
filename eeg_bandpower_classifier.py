pip install torch torchvision torchaudio
import argparse, json, math, os, random
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

DEFAULT_LABEL_MAP = {0: "delta", 1: "theta", 2: "alpha", 3: "beta", 4: "gamma"}

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# --------------------- Dataset ---------------------
class BandPowerDataset(Dataset):
    def __init__(self, X_bp: np.ndarray, y: np.ndarray, augment: bool=False, noise_std: float=0.03):
        assert X_bp.ndim == 3 and X_bp.shape[-1] == 5, "X_bp must be [N,C,5] with bands [d,t,a,b,g]"
        assert X_bp.shape[0] == y.shape[0]
        self.X = X_bp.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)
        self.augment = augment; self.noise_std = noise_std
        # log-scaled powers tend to help
        self.X = np.log1p(np.maximum(self.X, 0.0))
        # per-feature standardization
        mu = self.X.mean(axis=0, keepdims=True); sd = self.X.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-8, 1.0, sd); self.X = (self.X - mu)/sd

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx: int):
        x = self.X[idx]
        if self.augment:
            x = x + np.random.normal(0, self.noise_std, size=x.shape).astype(np.float32)
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long)

# ---------------------- Model ----------------------
class BPClassifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                     # [C,5] -> [C*5]
            nn.Linear(in_features, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
    def forward(self, x): return self.net(x)

# ------------------ Demo generation ----------------
def synth_band_signal(batch:int, channels:int, length:int, fs:int, band:Tuple[float,float]) -> np.ndarray:
    low, high = band; t = np.arange(length)/fs
    out = np.zeros((batch, channels, length), dtype=np.float32)
    for b in range(batch):
        freqs = np.random.uniform(low, high, size=(channels, 2))
        phases = np.random.uniform(0, 2*np.pi, size=(channels,2))
        amps = np.random.uniform(0.7, 1.3, size=(channels,2))
        for c in range(channels):
            sig = np.zeros_like(t, dtype=np.float32)
            for k in range(2):
                sig += (amps[c,k]*np.sin(2*np.pi*freqs[c,k]*t + phases[c,k])).astype(np.float32)
            sig += np.random.normal(0, 0.1, size=length).astype(np.float32)
            out[b,c]=sig
    return out

def bandpowers_from_timeseries(X: np.ndarray, fs:int, bands:Dict[int,Tuple[float,float]]) -> np.ndarray:
    # X: [N,C,T]; returns [N,C,5] in band order [d,t,a,b,g]
    N,C,T = X.shape
    freqs = np.fft.rfftfreq(T, d=1.0/fs)
    bp = np.zeros((N,C,5), dtype=np.float32)
    for i in range(N):
        Xf = np.abs(np.fft.rfft(X[i], axis=-1))**2 / T
        for bi,(lo,hi) in enumerate([bands[0],bands[1],bands[2],bands[3],bands[4]]):
            mask = (freqs >= lo) & (freqs < hi)
            # integrate power in the band
            bp[i,:,bi] = Xf[:,mask].mean(axis=1) if mask.any() else 0.0
    return bp

def make_demo_dataset(N:int=3000, C:int=4, T:int=512, fs:int=128):
    bands = {0:(0.5,4.0),1:(4.0,8.0),2:(8.0,12.0),3:(13.0,30.0),4:(30.0,45.0)}
    per = N//5; X_list=[]; y_list=[]
    for cls,band in bands.items():
        Xc = synth_band_signal(per, C, T, fs, band); X_list.append(Xc); y_list.append(np.full((per,),cls,np.int64))
    X = np.concatenate(X_list, axis=0); y = np.concatenate(y_list, axis=0)
    idx = np.random.permutation(X.shape[0]); X=X[idx]; y=y[idx]
    X_bp = bandpowers_from_timeseries(X, fs, bands)  # [N,C,5]
    return X_bp.astype(np.float32), y.astype(np.int64)

# ---------------- Train / Eval / Predict -----------
def train_one_epoch(model, loader, opt, device):
    model.train(); total=0.0; yT=[]; yP=[]
    for xb,yb in loader:
        xb=xb.to(device); yb=yb.to(device)
        opt.zero_grad(); logits=model(xb); loss=F.cross_entropy(logits,yb)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        total += loss.item()*xb.size(0)
        yT += yb.detach().cpu().tolist(); yP += logits.argmax(1).detach().cpu().tolist()
    n = len(loader.dataset); return total/n, accuracy_score(yT,yP), f1_score(yT,yP,average="macro")

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); total=0.0; yT=[]; yP=[]
    for xb,yb in loader:
        xb=xb.to(device); yb=yb.to(device); logits=model(xb); loss=F.cross_entropy(logits,yb)
        total += loss.item()*xb.size(0)
        yT += yb.detach().cpu().tolist(); yP += logits.argmax(1).detach().cpu().tolist()
    n=len(loader.dataset)
    return total/n, accuracy_score(yT,yP), f1_score(yT,yP,average="macro"), np.array(yT), np.array(yP)

@torch.no_grad()
def predict_one(bp_npy: str, ckpt: str, labels_json: str, threshold: float=0.6):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt, map_location=device)
    C = ck["in_channels"]; num_classes=ck["num_classes"]
    model = BPClassifier(in_features=C*5, num_classes=num_classes).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    with open(labels_json,"r",encoding="utf-8") as f: label_map = {int(k):v for k,v in json.load(f).items()}
    x = np.load(bp_npy).astype(np.float32)   # [C,5]
    assert x.ndim==2 and x.shape[1]==5, "Expected [C,5] bandpower array."
    x = np.log1p(np.maximum(x,0.0))
    # quick standardization using per-feature mean/std from sample itself (no train stats available)
    mu=x.mean(axis=0,keepdims=True); sd=x.std(axis=0,keepdims=True); sd=np.where(sd<1e-8,1.0,sd); x=(x-mu)/sd
    xt = torch.from_numpy(x).unsqueeze(0).to(device) # [1,C,5]
    probs = torch.softmax(model(xt), dim=1).squeeze(0).cpu().numpy()
    idx = int(np.argmax(probs)); conf=float(probs[idx]); label=label_map.get(idx,str(idx))
    return {"pred_index":idx,"pred_label":label,"confidence":round(conf,4),"is_known":bool(conf>=threshold),
            "threshold":threshold,"probs":{label_map.get(i,str(i)):float(p) for i,p in enumerate(probs)}}

def main():
    ap = argparse.ArgumentParser(description="EEG band-power classifier (PyTorch)")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--data", type=str, default=None, help="NPZ with X_bp:[N,C,5], y:[N]")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--test-split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint", type=str, default="bp_model_best.pth")
    ap.add_argument("--labels", type=str, default="bp_label_map.json")
    ap.add_argument("--predict-npy", type=str, default=None, help="Path to one [C,5] band-power window")
    ap.add_argument("--threshold", type=float, default=0.6)
    args = ap.parse_args(); set_seed(args.seed)

    if args.predict_npy is not None:
        out = predict_one(args.predict_npy, args.checkpoint, args.labels, args.threshold)
        print(json.dumps(out, indent=2)); return

    if args.demo:
        X_bp, y = make_demo_dataset(N=3000, C=4, T=512, fs=128)
        label_map = DEFAULT_LABEL_MAP.copy()
        print(f"[demo] X_bp: {X_bp.shape}, y: {y.shape}")
    else:
        assert args.data is not None, "Provide --data NPZ or use --demo"
        npz = np.load(args.data)
        X_bp = npz["X_bp"].astype(np.float32); y = npz["y"].astype(np.int64)
        if "label_map_json" in npz.files:
            label_map = json.loads(str(npz["label_map_json"].tolist()))
            label_map = {int(k): v for k, v in label_map.items()}
        else:
            classes = sorted(list(set(int(c) for c in np.unique(y).tolist())))
            label_map = {i: DEFAULT_LABEL_MAP.get(i, str(i)) for i in classes}
        print(f"[data] X_bp: {X_bp.shape}, y: {y.shape}")

    N, C, _ = X_bp.shape; num_classes = len(sorted(set(y.tolist())))
    val_n = int(math.floor(args.val_split * N)); test_n = int(math.floor(args.test_split * N))
    train_n = N - val_n - test_n; assert train_n>0 and val_n>0 and test_n>0
    idx = np.random.permutation(N); train_idx = idx[:train_n]; val_idx = idx[train_n:train_n+val_n]; test_idx = idx[train_n+val_n:]
    Xtr, ytr = X_bp[train_idx], y[train_idx]; Xva, yva = X_bp[val_idx], y[val_idx]; Xte, yte = X_bp[test_idx], y[test_idx]

    ds_tr = BandPowerDataset(Xtr, ytr, augment=True)
    ds_va = BandPowerDataset(Xva, yva, augment=False)
    ds_te = BandPowerDataset(Xte, yte, augment=False)
    tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0)
    va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=0)
    te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BPClassifier(in_features=C*5, num_classes=num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2, verbose=True)

    best_va = -1.0; best_state=None
    print(f"Device: {device} | Train: {len(ds_tr)} | Val: {len(ds_va)} | Test: {len(ds_te)}")
    for ep in range(1, args.epochs+1):
        tl, ta, tf = train_one_epoch(model, tr, opt, device)
        vl, vaa, vf, yv, pv = evaluate(model, va, device); scheduler.step(vl)
        print(f"Epoch {ep:02d}/{args.epochs} | Train loss {tl:.4f} acc {ta:.3f} f1 {tf:.3f} || Val loss {vl:.4f} acc {vaa:.3f} f1 {vf:.3f}")
        if vaa > best_va:
            best_va = vaa
            best_state = {"state_dict": model.state_dict(), "in_channels": C, "num_classes": num_classes, "epoch": ep, "val_acc": best_va}

    if best_state is None:
        best_state = {"state_dict": model.state_dict(), "in_channels": C, "num_classes": num_classes, "epoch": args.epochs, "val_acc": best_va}
    torch.save(best_state, args.checkpoint)
    with open(args.labels, "w", encoding="utf-8") as f: json.dump({int(k):v for k,v in label_map.items()}, f, indent=2)
    print(f"Saved: {args.checkpoint}, {args.labels}")

    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["state_dict"])
    tl, ta, tf, yt, pt = evaluate(model, te, device)
    print(f"[Test] loss {tl:.4f} acc {ta:.3f} f1 {tf:.3f}")
    print("Confusion matrix (rows=true, cols=pred):"); print(confusion_matrix(yt, pt))
    target_names = [label_map.get(i, str(i)) for i in sorted(set(yt.tolist()))]
    print(classification_report(yt, pt, target_names=target_names, digits=3))

if __name__ == "__main__":
    main()
