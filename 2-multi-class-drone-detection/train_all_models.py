"""
Kullanım
--------
  # 1) Drone tanıma (binary), ana deney
  python train_all_models.py --task binary

  # 2) 3 sınıflı (bebop / membo / unknown); etiket dosya adından türetilir
  python train_all_models.py --task multiclass

  # 3) Öznitelik çıkarım süresini de ölçmek için ses dosyalarının klasörü:
  python train_all_models.py --task binary --audio_dir ./drone_audio_wavs

  # 4) Dış veri setinden öznitelik CSV'si üretme (aynı öznitelik tanımıyla):
  python train_all_models.py --extract ./dis_veri/drone  yes_drone  dis_drone.csv
  python train_all_models.py --extract ./dis_veri/diger  unknown    dis_unknown.csv
  #    (iki CSV'yi tek dosyada birleştirip) --external_csv dis_veri.csv ile kullanın

  # 5) Birden çok farklı bölme (rastgelelik) ile ortalama ± std:
  python train_all_models.py --task binary --seeds 42 43 44 45 46

Kurulum:
  pip install pandas numpy scikit-learn xgboost lightgbm matplotlib librosa soundfile psutil

"""

import argparse
import gc
import io
import json
import os
import re
import tempfile
import time
import warnings

warnings.filterwarnings("ignore")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_recall_fscore_support, precision_score,
    recall_score, classification_report,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("UYARI: xgboost kurulu değil, atlanacak (pip install xgboost)")

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("UYARI: lightgbm kurulu değil, atlanacak (pip install lightgbm)")

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # çok nadir; sklearn bağımlılığıdır
    threadpool_limits = None

try:
    import psutil
except ImportError:
    psutil = None


# ----------------------------------------------------------------------
# SABİTLER  (makalede AYNEN raporlanacak değerler)
# ----------------------------------------------------------------------
SR = 22050              # örnekleme frekansı (Hz), mono
N_FFT = 2048            # FFT penceresi (örnek)  -> ~92.9 ms @22.05 kHz
WIN_LENGTH = 2048       # pencere uzunluğu (= n_fft)
HOP_LENGTH = 512        # kaydırma (örnek)       -> ~23.2 ms, %75 örtüşme
WINDOW = "hann"         # pencere fonksiyonu
CENTER = True
PAD_MODE = "constant"
N_MFCC = 13
N_MELS = 128
DELTA_WIDTH = 9
DELTA_MODE = "interp"
N_CHROMA = 12
N_CONTRAST_BANDS = 6    # -> 7 spektral kontrast değeri (6 bant + 1)
CONTRAST_FMIN = 200.0
ROLLOFF_PERCENT = 0.85
FIXED_LENGTH_SEC = 1.0  # extract_audio_features.py / droneses.py ile AYNI olmalı (None = kapalı)
CROP_MODE = "center"
STFT_KW = dict(n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
               window=WINDOW, center=CENTER, pad_mode=PAD_MODE)

# Modele GİRMEYEN sütunlar
META_COLS = ["filename", "label_name", "label"]
DROP_FEATURES = ["duration_sec"]   # H1: kayıt uzunluğu artefaktı

TEST_SIZE_FOLDS = 5     # StratifiedGroupKFold(5) -> test = 1 fold ≈ %20
CV_FOLDS = 5
RANDOM_STATE = 42
BENCH_REPEATS = 300     # çıkarım gecikmesi tekrar sayısı
BENCH_WARMUP = 20


# ----------------------------------------------------------------------
# ÖZNİTELİK ÇIKARIMI (H4: tüm pencere parametreleri açık)
# ----------------------------------------------------------------------
def feature_names():
    base = ["zcr", "rms", "spec_centroid", "spec_bandwidth",
            "spec_rolloff", "spec_flatness"]
    base += [f"spec_contrast_{i}" for i in range(N_CONTRAST_BANDS + 1)]
    base += [f"mfcc_{i}" for i in range(N_MFCC)]
    base += [f"mfcc_delta_{i}" for i in range(N_MFCC)]
    base += [f"chroma_{i}" for i in range(N_CHROMA)]
    names = []
    for b in base:
        names += [f"{b}_mean", f"{b}_std"]
    return names


def prepare_signal(y, sr=SR):
    """Klibi FIXED_LENGTH_SEC uzunluğuna getirir (kırp / döngüsel tekrar ile tamamla)."""
    if FIXED_LENGTH_SEC is None:
        return y
    n = int(round(FIXED_LENGTH_SEC * sr))
    if len(y) > n:
        start = (len(y) - n) // 2 if CROP_MODE == "center" else 0
        y = y[start:start + n]
    elif len(y) < n:
        y = np.pad(y, (0, n - len(y)), mode="wrap")
    return y


def extract_features_from_signal(y, sr=SR):
    """
    102 boyutlu öznitelik vektörü. extract_audio_features.py / droneses.py içindeki
    fonksiyonla BİREBİR aynı librosa çağrılarını kullanır (duration_sec yok).
    """
    import librosa
    y = prepare_signal(y, sr)
    rows = []
    rows.append(librosa.feature.zero_crossing_rate(
        y, frame_length=N_FFT, hop_length=HOP_LENGTH, center=CENTER))
    rows.append(librosa.feature.rms(
        y=y, frame_length=N_FFT, hop_length=HOP_LENGTH, center=CENTER, pad_mode=PAD_MODE))
    rows.append(librosa.feature.spectral_centroid(y=y, sr=sr, **STFT_KW))
    rows.append(librosa.feature.spectral_bandwidth(y=y, sr=sr, **STFT_KW))
    rows.append(librosa.feature.spectral_rolloff(
        y=y, sr=sr, roll_percent=ROLLOFF_PERCENT, **STFT_KW))
    rows.append(librosa.feature.spectral_flatness(y=y, **STFT_KW))
    rows.append(librosa.feature.spectral_contrast(
        y=y, sr=sr, n_bands=N_CONTRAST_BANDS, fmin=CONTRAST_FMIN, **STFT_KW))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_mels=N_MELS, **STFT_KW)
    rows.append(mfcc)
    rows.append(librosa.feature.delta(mfcc, width=DELTA_WIDTH, order=1, mode=DELTA_MODE))
    rows.append(librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA, **STFT_KW))
    vec = []
    for r in rows:
        for k in range(r.shape[0]):
            vec += [float(np.mean(r[k])), float(np.std(r[k]))]
    return np.asarray(vec, dtype=np.float64)


def extract_features_from_file(path):
    import librosa
    y, sr = librosa.load(path, sr=SR, mono=True)
    return extract_features_from_signal(y, sr)


def extract_folder_to_csv(folder, label_name, out_csv):
    """Dış veri seti klasöründen (wav) features.csv ile aynı şemada CSV üretir."""
    files = sorted(f for f in os.listdir(folder)
                   if f.lower().endswith((".wav", ".flac", ".ogg", ".mp3")))
    if not files:
        raise SystemExit(f"'{folder}' içinde ses dosyası bulunamadı.")
    names = feature_names()
    recs = []
    for i, f in enumerate(files, 1):
        try:
            v = extract_features_from_file(os.path.join(folder, f))
        except Exception as e:
            print(f"  [ATLANDI] {f}: {e}")
            continue
        recs.append([f, label_name, 0 if label_name == "unknown" else 1, *v])
        if i % 200 == 0:
            print(f"  {i}/{len(files)}")
    df = pd.DataFrame(recs, columns=["filename", "label_name", "label"] + names)
    df.to_csv(out_csv, index=False)
    print(f"{len(df)} kayıt -> {out_csv} (öznitelik boyutu: {len(names)})")


# ----------------------------------------------------------------------
# VERİ, ETİKET VE GRUP (H1/H2)
# ----------------------------------------------------------------------
def record_group(filename):
    """
    Klibin geldiği ORİJİNAL KAYDI belirleyen grup kimliği.
      drone:   'B_S2_D1_067-bebop_003_.wav' -> 'B_S2_D1_067-bebop'
      ESC-50:  '4-147658-A-464.wav'         -> 'esc_147658'  (freesound klip no)
      diğer:   'doing_the_dishes012.wav'    -> 'doing_the_dishes'
    Aynı gruptaki klipler asla hem eğitimde hem testte bulunmaz.
    NOT: Dosya adı oturum (session) bilgisi içermiyorsa grup düzeyi 'kayıt'tır;
    makalede "kayıt bazlı (record-wise)" olarak belirtilmelidir.
    """
    m = re.match(r"^(\d)-(\d+)-[A-Za-z]-\d+", filename)
    if m:
        return "esc_" + m.group(2)
    m = re.match(r"^(.*)-(bebop|membo)_\d{3}_\.wav$", filename, re.I)
    if m:
        return m.group(1) + "-" + m.group(2).lower()
    if filename.startswith("extra_"):
        return filename
    if re.match(r"^(BACKGROUND|DRONE|HELICOPTER)_\d+\.wav$", filename, re.I):
        return filename   # her dosya ayrı kayıt
    return re.sub(r"\d+\.[A-Za-z0-9]+$", "", filename)


def make_labels(df, task):
    """task='binary' -> unknown / yes_drone ; 'multiclass' -> unknown / bebop / membo"""
    if task == "binary":
        classes = ["unknown", "yes_drone"]
        is_drone = df["label_name"].astype(str).isin(["background", "drone", "helicopter"])
        lab = np.where(is_drone, "yes_drone", "unknown")
        mask = np.ones(len(df), bool)
    else:
        # Bu CSV background/drone/helicopter etiketlerini içeriyorsa 3 sınıflı görev bunlar üzerinden kurulur
        present = set(df["label_name"].astype(str))
        if {"background", "drone", "helicopter"} <= present:
            classes = ["background", "drone", "helicopter"]
            lab = df["label_name"].astype(str).values
            mask = np.ones(len(df), bool)
            y = np.array([classes.index(l) for l in lab])
            return y, classes, mask
        classes = ["unknown", "bebop", "membo"]
        out, mask = [], []
        for fn, ln in zip(df["filename"], df["label_name"].astype(str)):
            if ln == "unknown":
                out.append("unknown"); mask.append(True)
            elif ln in ("bebop", "membo"):
                out.append(ln); mask.append(True)
            elif "membo" in fn.lower():
                out.append("membo"); mask.append(True)
            elif "bebop" in fn.lower():
                out.append("bebop"); mask.append(True)
            else:   # model türü belirlenemeyen drone kaydı
                out.append("?"); mask.append(False)
        lab, mask = np.array(out), np.array(mask)
        if (~mask).sum():
            print(f"UYARI: {(~mask).sum()} drone kaydının modeli (bebop/membo) "
                  f"belirlenemedi, 3 sınıflı görevden çıkarıldı.")
    y = np.array([classes.index(l) if l in classes else -1 for l in lab])
    return y, classes, mask


def load_csv(path, task):
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c not in META_COLS + DROP_FEATURES]
    y, classes, mask = make_labels(df, task)
    df = df[mask].reset_index(drop=True)
    y = y[mask]
    groups = df["filename"].map(record_group).values
    return df, df[feat_cols].values, y, groups, feat_cols, classes


# ----------------------------------------------------------------------
# MODELLER (H5: hiperparametreler açık) ve KAYIP FONKSİYONLARI
# ----------------------------------------------------------------------
def get_models(task, n_classes):
    models = {
        "Logistic Regression": LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE),
        "SVM (Linear)": SVC(kernel="linear", C=1.0, random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, weights="uniform",
                                                    metric="minkowski", p=2),
        "Decision Tree": DecisionTreeClassifier(criterion="gini", max_depth=None,
                                                random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, criterion="gini",
                                                max_depth=None, random_state=RANDOM_STATE),
        "Extra Trees": ExtraTreesClassifier(n_estimators=300, criterion="gini",
                                            max_depth=None, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, loss="log_loss",
            random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, learning_rate=1.0,
                                       random_state=RANDOM_STATE),
        "Naive Bayes": GaussianNB(var_smoothing=1e-9),
        "Linear Discriminant Analysis": LinearDiscriminantAnalysis(solver="svd"),
        "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                                          solver="adam", alpha=1e-4, learning_rate_init=1e-3,
                                          max_iter=1000, random_state=RANDOM_STATE),
    }
    if XGB_AVAILABLE:
        obj = "binary:logistic" if n_classes == 2 else "multi:softprob"
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, learning_rate=0.3, max_depth=6, objective=obj,
            eval_metric="logloss" if n_classes == 2 else "mlogloss",
            reg_lambda=1.0, n_jobs=1, random_state=RANDOM_STATE, verbosity=0)
    if LGBM_AVAILABLE:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300, learning_rate=0.1, num_leaves=31, max_depth=-1,
            objective="binary" if n_classes == 2 else "multiclass",
            reg_lambda=0.0, n_jobs=1, random_state=RANDOM_STATE, verbose=-1)
    return models


# Her modelin eğitimde optimize ettiği kayıp / ölçüt (makaleye tablo olarak eklenebilir)
LOSS_INFO = {
    "Logistic Regression": "Log-loss (çapraz entropi) + L2 düzenlileştirme (C=1.0)",
    "SVM (RBF)": "Hinge loss + L2 marjin düzenlileştirme (C=1.0); çok sınıfta one-vs-one",
    "SVM (Linear)": "Hinge loss + L2 marjin düzenlileştirme (C=1.0); çok sınıfta one-vs-one",
    "K-Nearest Neighbors": "Kayıp fonksiyonu yok (örnek tabanlı); Öklid uzaklığı, k=5",
    "Decision Tree": "Gini safsızlığı azaltımı (bölünme ölçütü)",
    "Random Forest": "Gini safsızlığı azaltımı (bölünme ölçütü), 300 ağaç, bagging",
    "Extra Trees": "Gini safsızlığı azaltımı (rastgele eşik), 300 ağaç",
    "Gradient Boosting": "Log-loss (binomial/multinomial deviance), öğrenme oranı 0.1",
    "AdaBoost": "Üstel kayıp (SAMME), 50 zayıf öğrenici (derinlik-1 ağaç)",
    "Naive Bayes": "Kayıp yok; maksimum olabilirlik (Gaussian olasılıksal model)",
    "Linear Discriminant Analysis": "Kayıp yok; Gauss varsayımlı Bayes karar kuralı (SVD çözücü)",
    "MLP (Neural Net)": "Log-loss (çapraz entropi) + L2 (alpha=1e-4), Adam",
    "XGBoost": "Log-loss (logistic / softmax) + L2 (lambda=1.0)",
    "LightGBM": "Log-loss (binary / multiclass softmax çapraz entropi)",
    "Dummy (çoğunluk sınıfı)": "Yok; her zaman en sık sınıfı tahmin eder",
}


def make_pipeline(model):
    # Ölçekleyici Pipeline içinde: her CV katında yalnızca eğitim kısmına fit edilir
    return Pipeline([("scaler", StandardScaler()), ("clf", clone(model))])


def write_hyperparameter_table(models, path):
    rows = []
    for name, m in models.items():
        params = {k: v for k, v in m.get_params().items()}
        rows.append({
            "Model": name,
            "Kayıp / optimize edilen ölçüt": LOSS_INFO.get(name, ""),
            "Hiperparametreler (tam liste)": json.dumps(params, default=str, ensure_ascii=False),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


# ----------------------------------------------------------------------
# METRİKLER (H3: macro VE weighted)
# ----------------------------------------------------------------------
def compute_metrics(y_true, y_pred, classes):
    labels = list(range(len(classes)))
    m = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Acc.": balanced_accuracy_score(y_true, y_pred),
        "Precision (macro)": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1 (macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "F1 (weighted)": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    for i, c in enumerate(classes):          # sınıf bazlı (azınlık sınıfları görmek için)
        m[f"Precision [{c}]"] = p[i]
        m[f"Recall [{c}]"] = r[i]
        m[f"F1 [{c}]"] = f[i]
    if len(classes) == 2:                    # binary: pozitif = drone
        m["FN (kaçan drone)"] = int(((y_pred == 0) & (y_true == 1)).sum())
        m["FP (yanlış alarm)"] = int(((y_pred == 1) & (y_true == 0)).sum())
    return m


# ----------------------------------------------------------------------
# HAFİFLİK / GERÇEK ZAMAN ÖLÇÜMLERİ (H6)
# ----------------------------------------------------------------------
def model_size_kb(pipe):
    buf = io.BytesIO()
    joblib.dump(pipe, buf, compress=0)
    return buf.getbuffer().nbytes / 1024.0


def ram_delta_mb(pipe, x1):
    """Modeli diskten yükleyip bir tahmin yaparken süreç RSS artışı (yaklaşık, MB)."""
    if psutil is None:
        return np.nan
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.joblib")
        joblib.dump(pipe, p, compress=0)
        gc.collect()
        proc = psutil.Process(os.getpid())
        r0 = proc.memory_info().rss
        loaded = joblib.load(p)
        loaded.predict(x1)
        r1 = proc.memory_info().rss
        del loaded
        gc.collect()
    return max(r1 - r0, 0) / (1024 ** 2)


def inference_latency_ms(pipe, X):
    """Tek örnek (batch=1), tek iş parçacığı: medyan ve p95 (ms)."""
    idx = np.random.RandomState(0).randint(0, len(X), BENCH_REPEATS + BENCH_WARMUP)
    ctx = threadpool_limits(limits=1) if threadpool_limits else None
    try:
        for i in idx[:BENCH_WARMUP]:
            pipe.predict(X[i:i + 1])
        ts = []
        for i in idx[BENCH_WARMUP:]:
            t0 = time.perf_counter()
            pipe.predict(X[i:i + 1])
            ts.append((time.perf_counter() - t0) * 1000.0)
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
    return float(np.median(ts)), float(np.percentile(ts, 95))


def feature_extraction_ms(audio_dir, n_files=100):
    """Ses dosyalarından öznitelik çıkarım süresi (ms/örnek): yükleme dahil / hariç."""
    import librosa
    files = sorted(f for f in os.listdir(audio_dir)
                   if f.lower().endswith((".wav", ".flac", ".ogg", ".mp3")))[:n_files]
    if not files:
        return None
    # ısınma (numba/JIT)
    extract_features_from_file(os.path.join(audio_dir, files[0]))
    t_total, t_feat, dur = [], [], []
    for f in files:
        p = os.path.join(audio_dir, f)
        t0 = time.perf_counter()
        y, sr = librosa.load(p, sr=SR, mono=True)
        t1 = time.perf_counter()
        extract_features_from_signal(y, sr)
        t2 = time.perf_counter()
        t_total.append((t2 - t0) * 1000.0)
        t_feat.append((t2 - t1) * 1000.0)
        dur.append(len(y) / sr)
    return {"n": len(files), "mean_clip_sec": float(np.mean(dur)),
            "feat_ms_median": float(np.median(t_feat)),
            "load_plus_feat_ms_median": float(np.median(t_total))}


# ----------------------------------------------------------------------
# ANA AKIŞ
# ----------------------------------------------------------------------
def run_one_seed(seed, X, y, groups, classes, models, rows_cv, rows_test, preds_store):
    # --- Kayıt bazlı %80/%20 bağımsız test ayrımı ---
    sgkf = StratifiedGroupKFold(n_splits=TEST_SIZE_FOLDS, shuffle=True, random_state=seed)
    tr_idx, te_idx = next(sgkf.split(X, y, groups))
    assert set(groups[tr_idx]).isdisjoint(groups[te_idx]), "Eğitim/test grup kesişimi var!"
    Xtr, Xte, ytr, yte = X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]
    gtr = groups[tr_idx]

    if seed == RANDOM_STATE or not preds_store.get("info"):
        info = {
            "train_n": len(tr_idx), "test_n": len(te_idx),
            "train_groups": len(set(gtr)), "test_groups": len(set(groups[te_idx])),
            "train_class_counts": dict(zip(*np.unique(ytr, return_counts=True))),
            "test_class_counts": dict(zip(*np.unique(yte, return_counts=True))),
        }
        preds_store["info"] = info
        print(f"  Eğitim: {info['train_n']} klip / {info['train_groups']} kayıt | "
              f"Test: {info['test_n']} klip / {info['test_groups']} kayıt "
              f"(kayıtlar ayrık - doğrulandı)")
        print(f"  Sınıf dağılımı  eğitim: {info['train_class_counts']}  "
              f"test: {info['test_class_counts']}")

    cv = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)

    # --- Dummy baseline ---
    dummy = make_pipeline(DummyClassifier(strategy="most_frequent"))
    dummy.fit(Xtr, ytr)
    mt = compute_metrics(yte, dummy.predict(Xte), classes)
    rows_test.append({"Seed": seed, "Model": "Dummy (çoğunluk sınıfı)", **mt})

    for name, model in models.items():
        pipe = make_pipeline(model)
        try:
            # CV: yalnızca EĞİTİM kümesi, kayıt bazlı katlar
            cvres = cross_validate(
                pipe, Xtr, ytr, cv=cv, groups=gtr,
                scoring={"acc": "accuracy", "f1m": "f1_macro", "bacc": "balanced_accuracy"},
                n_jobs=1)
            t0 = time.perf_counter()
            pipe.fit(Xtr, ytr)
            fit_s = time.perf_counter() - t0
            y_pred = pipe.predict(Xte)
            mt = compute_metrics(yte, y_pred, classes)
            rows_cv.append({
                "Seed": seed, "Model": name,
                "CV Acc (mean)": cvres["test_acc"].mean(), "CV Acc (std)": cvres["test_acc"].std(),
                "CV F1-macro (mean)": cvres["test_f1m"].mean(), "CV F1-macro (std)": cvres["test_f1m"].std(),
                "CV Bal.Acc (mean)": cvres["test_bacc"].mean(),
            })
            rows_test.append({"Seed": seed, "Model": name, "Train time (s)": fit_s, **mt})
            preds_store.setdefault((seed, "pred"), {})[name] = y_pred
            preds_store[(seed, "yte")] = yte
            preds_store.setdefault((seed, "pipe"), {})[name] = pipe
            preds_store[(seed, "Xte")] = Xte
            print(f"  [OK] {name:<30} Test Acc={mt['Accuracy']:.4f} | "
                  f"F1-macro={mt['F1 (macro)']:.4f} | F1-weighted={mt['F1 (weighted)']:.4f} | "
                  f"CV Acc={cvres['test_acc'].mean():.4f} (±{cvres['test_acc'].std():.4f})")
        except Exception as e:
            print(f"  [HATA] {name}: {e}")


def plot_confusions(preds, seed, classes, path):
    ypred = preds[(seed, "pred")]
    yte = preds[(seed, "yte")]
    n = len(ypred)
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows))
    axes = np.array(axes).reshape(-1)
    for ax, (name, yp) in zip(axes, ypred.items()):
        cm = confusion_matrix(yte, yp, labels=list(range(len(classes))))
        ax.imshow(cm, cmap="Blues")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek")
        ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(classes, fontsize=8)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def benchmark_table(preds, seed, audio_dir):
    """H6: model boyutu, gecikme, RAM, eğitim süresi (+ varsa öznitelik çıkarım süresi)."""
    pipes = preds[(seed, "pipe")]
    Xte = preds[(seed, "Xte")]
    fe = None
    if audio_dir:
        print(f"\nÖznitelik çıkarım süresi ölçülüyor ('{audio_dir}')...")
        fe = feature_extraction_ms(audio_dir)
        if fe:
            print(f"  {fe['n']} klip, ort. süre {fe['mean_clip_sec']:.2f} s | "
                  f"öznitelik: {fe['feat_ms_median']:.2f} ms/örnek (medyan) | "
                  f"yükleme+öznitelik: {fe['load_plus_feat_ms_median']:.2f} ms/örnek")
    rows = []
    for name, pipe in pipes.items():
        med, p95 = inference_latency_ms(pipe, Xte)
        row = {
            "Model": name,
            "Model boyutu (KB)": round(model_size_kb(pipe), 1),
            "Çıkarım gecikmesi medyan (ms)": round(med, 3),
            "Çıkarım gecikmesi p95 (ms)": round(p95, 3),
            "RAM artışı (MB, yaklaşık)": round(ram_delta_mb(pipe, Xte[:1]), 2),
            "Öznitelik çıkarım (ms/örnek)": round(fe["feat_ms_median"], 2) if fe else np.nan,
            "Toplam (öznitelik+çıkarım, ms)": round(fe["feat_ms_median"] + med, 2) if fe else np.nan,
        }
        rows.append(row)
        print(f"  {name:<30} boyut={row['Model boyutu (KB)']:>9.1f} KB | "
              f"gecikme={med:.3f} ms (p95 {p95:.3f}) | RAM+={row['RAM artışı (MB, yaklaşık)']} MB")
    return pd.DataFrame(rows)


def evaluate_external(args, X, y, feat_cols, classes, models):
    """Ana veriyle eğit (tüm kayıtlar), dış veri setinde test et (H7)."""
    ext = pd.read_csv(args.external_csv)
    missing = [c for c in feat_cols if c not in ext.columns]
    if missing:
        raise SystemExit(f"Dış CSV'de eksik öznitelik sütunları: {missing[:5]}...")
    yext, _, mask = make_labels(ext, args.task)
    ext, yext = ext[mask].reset_index(drop=True), yext[mask]
    Xext = ext[feat_cols].values
    print(f"\nDış değerlendirme: {len(ext)} örnek | sınıf dağılımı: "
          f"{dict(zip(*np.unique(yext, return_counts=True)))}")
    rows = []
    for name, model in models.items():
        pipe = make_pipeline(model).fit(X, y)
        rows.append({"Model": name, **compute_metrics(yext, pipe.predict(Xext), classes)})
        print(f"  {name:<30} Acc={rows[-1]['Accuracy']:.4f} F1-macro={rows[-1]['F1 (macro)']:.4f}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Drone ses sınıflandırma (revize)")
    ap.add_argument("--csv", default="features.csv")
    ap.add_argument("--task", choices=["binary", "multiclass"], default="multiclass")
    ap.add_argument("--seeds", type=int, nargs="+", default=[RANDOM_STATE])
    ap.add_argument("--audio_dir", default=None,
                    help="Öznitelik çıkarım süresini ölçmek için wav klasörü")
    ap.add_argument("--external_csv", default=None,
                    help="Dış veri seti öznitelik CSV'si (aynı şema)")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--extract", nargs=3, metavar=("KLASÖR", "ETİKET", "ÇIKIŞ_CSV"),
                    help="Klasördeki seslerden öznitelik CSV'si üret ve çık")
    args = ap.parse_args()

    if args.extract:
        extract_folder_to_csv(*args.extract)
        return

    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.task

    # ---------------- Veri ----------------
    df, X, y, groups, feat_cols, classes = load_csv(args.csv, args.task)
    print("=" * 90)
    print(f"Görev: {args.task} | sınıflar: {classes}")
    print(f"Örnek sayısı: {len(y)} | kayıt (grup) sayısı: {len(set(groups))}")
    print(f"Sınıf dağılımı: { {classes[k]: int(v) for k, v in zip(*np.unique(y, return_counts=True))} }")
    print(f"Nihai öznitelik vektörü boyutu: {len(feat_cols)} "
          f"(= {len(feat_cols)//2} öznitelik × [ortalama, std])")
    print(f"Modele girmeyen sütunlar: {DROP_FEATURES} (kayıt uzunluğu artefaktı)")
    print(f"Librosa parametreleri: sr={SR} Hz (mono), n_fft={N_FFT}, win_length={WIN_LENGTH}, "
          f"hop_length={HOP_LENGTH}, window='{WINDOW}', n_mfcc={N_MFCC}, n_chroma={N_CHROMA}, "
          f"contrast_bands={N_CONTRAST_BANDS}")
    print(f"Bölme: kayıt bazlı StratifiedGroupKFold (test ≈ %{100//TEST_SIZE_FOLDS}), "
          f"CV: {CV_FOLDS} katlı kayıt bazlı, yalnızca eğitim kümesinde")
    print(f"Çoğunluk sınıfı baseline accuracy: {np.bincount(y).max() / len(y):.4f}")
    print("=" * 90)

    models = get_models(args.task, len(classes))
    write_hyperparameter_table(models, os.path.join(args.out_dir, f"model_hyperparameters_{tag}.csv"))

    # ---------------- Eğit / değerlendir ----------------
    rows_cv, rows_test, store = [], [], {}
    for s in args.seeds:
        print(f"\n--- Seed {s} ---")
        run_one_seed(s, X, y, groups, classes, models, rows_cv, rows_test, store)

    cv_df = pd.DataFrame(rows_cv)
    te_df = pd.DataFrame(rows_test)
    num_cols = [c for c in te_df.columns if c not in ("Seed", "Model")]
    te_mean = te_df.groupby("Model", sort=False)[num_cols].mean()
    te_std = te_df.groupby("Model", sort=False)[num_cols].std() if len(args.seeds) > 1 else None
    cv_mean = cv_df.groupby("Model", sort=False).mean(numeric_only=True).drop(columns="Seed")

    summary = te_mean.join(cv_mean, how="left")
    if te_std is not None:
        summary["Accuracy (std, seeds)"] = te_std["Accuracy"]
        summary["F1 (macro) (std, seeds)"] = te_std["F1 (macro)"]
    summary = summary.sort_values("F1 (macro)", ascending=False)
    summary.round(4).to_csv(os.path.join(args.out_dir, f"model_comparison_{tag}.csv"))

    show = ["Accuracy", "Balanced Acc.", "Precision (macro)", "Recall (macro)",
            "F1 (macro)", "F1 (weighted)", "MCC", "CV Acc (mean)", "CV Acc (std)"]
    print("\n" + "=" * 110)
    print(f"BAĞIMSIZ (kayıt bazlı) TEST KÜMESİ SONUÇLARI - {args.task} "
          f"(F1: macro ve weighted birlikte; {len(args.seeds)} seed ortalaması)")
    print("=" * 110)
    print(summary[show].round(4).to_string())

    best = summary.drop(index="Dummy (çoğunluk sınıfı)", errors="ignore").index[0]
    print(f"\nEn iyi model (test F1-macro): {best}")
    s0 = args.seeds[0]
    best_pred = store[(s0, "pred")][best]
    print(f"\n--- {best}: sınıf bazlı rapor (seed {s0}) ---")
    print(classification_report(store[(s0, "yte")], best_pred, target_names=classes, digits=4))

    plot_confusions(store, s0, classes, os.path.join(args.out_dir, f"confusion_matrices_{tag}.png"))

    # ---------------- Hafiflik tablosu ----------------
    print("\n" + "=" * 90)
    print("HAFİFLİK / GERÇEK ZAMAN ÖLÇÜMLERİ (bu bilgisayarda, tek iş parçacığı, batch=1)")
    print("=" * 90)
    bench = benchmark_table(store, s0, args.audio_dir)
    tr = te_df[te_df.Seed == s0][["Model", "Train time (s)"]].dropna()
    bench = bench.merge(tr, on="Model", how="left")
    bench.round(3).to_csv(os.path.join(args.out_dir, f"efficiency_{tag}.csv"), index=False)
    print("\nNOT: Bu değerler masaüstü bilgisayara aittir. Gömülü cihaz iddiası için aynı "
          "ölçümü hedef donanımda (örn. Raspberry Pi) tekrarlayın; enerji tüketimi ayrıca "
          "harici güç ölçerle ölçülmelidir (burada ölçülmemiştir).")

    # ---------------- Dış veri seti ----------------
    if args.external_csv:
        ext_df = evaluate_external(args, X, y, feat_cols, classes, models)
        ext_df.round(4).to_csv(os.path.join(args.out_dir, f"external_{tag}.csv"), index=False)

    print(f"\nTüm çıktılar '{args.out_dir}/' klasörüne yazıldı.")


if __name__ == "__main__":
    main()
