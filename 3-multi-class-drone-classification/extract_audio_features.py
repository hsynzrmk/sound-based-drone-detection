# -*- coding: utf-8 -*-
"""
Drone sesi sınıflandırması için 1 boyutlu (1D) ses özniteliği çıkarım betiği.
3 SINIFLI VERSİYON: bebop / membo / unknown

Klasör yapısı (script ile aynı yerde ya da yolunu ROOT_DIR'e yazarak belirtin):

    ana_klasor/
        bebop/
            ses1.wav
            ...
        membo/
            ses1.wav
            ...
        unknown/
            ses1.wav
            ...

Kurulum:
    pip install librosa numpy pandas soundfile tqdm

Çalıştırma:
    python extract_audio_features.py

Çıktı:
    features.csv                    -> her satır bir klibe ait 102 boyutlu öznitelik
                                       vektörü + filename, label_name, label sütunları
    features_extraction_info.json   -> kullanılan TÜM çıkarım parametreleri, kütüphane
                                       sürümleri ve öznitelik çıkarım süreleri (ms/örnek)

REVİZYON NOTLARI (hakem yorumlarına göre)
-----------------------------------------
1) `duration_sec` özniteliği KALDIRILDI. Veri setinde kayıt uzunluğu sınıflara göre
   sistematik olarak farklıdır (drone klipleri <= 1.024 s, unknown klipleri çoğunlukla
   1.022 / 0.975 / 0.960 s, bir kısmı 5 s); bu yüzden süre tek başına sınıfı
   neredeyse kusursuz ayırır ve drone'un akustik bir özelliği değildir.
2) FIXED_LENGTH_SEC (varsayılan 1.0 s): tüm klipler sabit uzunluğa getirilir.
   Uzun klipler ortadan kırpılır; kısa klipler sessizlikle DEĞİL, kendi içinde
   döngüsel tekrarla ('wrap') tamamlanır (sıfır dolgusu sınıfa özgü bir iz bırakırdı).
   Orijinal (kırpmasız) davranış için FIXED_LENGTH_SEC = None yapın.
3) Pencereleme parametreleri açıkça yazıldı. Önceki sürüm librosa VARSAYILANLARINI
   kullanıyordu; aşağıdaki değerler bu varsayılanlarla aynıdır (n_fft=2048,
   win_length=n_fft, hop_length=512, 'hann', center=True, pad_mode='constant',
   n_mels=128, delta width=9 / mode='interp'). Böylece makalede birebir raporlanabilir.
4) Öznitelik çıkarım süresi (yükleme hariç/dahil) ölçülüp JSON dosyasına yazılır.
   Not: bu süreler betiğin çalıştığı bilgisayara aittir.
"""

import json
import os
import platform
import time

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

# ----------------------------------------------------------------------
# AYARLAR
# ----------------------------------------------------------------------
ROOT_DIR = r"F:\Datasets\drone\DroneAudioDataset-master\Multiclass_Drone_Audio"            # ana klasörün yolu (gerekirse değiştirin)
# klasör adı -> sayısal etiket (3 sınıf)
CLASS_FOLDERS = {
    "bebop": 0,
    "membo": 1,
    "unknown": 2,
}
SAMPLE_RATE = 22050               # tüm dosyalar bu örnekleme hızına yeniden örneklenir (mono)
N_MFCC = 13
OUTPUT_CSV = "features.csv"
OUTPUT_INFO_JSON = "features_extraction_info.json"
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

# Sabit klip uzunluğu (saniye). None -> kırpma/doldurma yok (orijinal davranış).
FIXED_LENGTH_SEC = 1.0
CROP_MODE = "center"              # uzun klipler için: "center" veya "start"

# --- Pencereleme / spektral analiz parametreleri (makalede aynen verilmeli) ---
N_FFT = 2048                      # FFT boyutu (~92.9 ms @22.05 kHz)
WIN_LENGTH = 2048                 # pencere uzunluğu (= n_fft)
HOP_LENGTH = 512                  # atlama (~23.2 ms, %75 örtüşme)
WINDOW = "hann"                   # pencere fonksiyonu
CENTER = True                     # çerçeveler sinyal etrafında ortalanır
PAD_MODE = "constant"             # center=True için kenar dolgusu (librosa >= 0.10 varsayılanı)
N_MELS = 128                      # MFCC için mel filtre sayısı
DELTA_WIDTH = 9                   # MFCC delta pencere genişliği
DELTA_MODE = "interp"             # MFCC delta kenar işleme
CONTRAST_BANDS = 6                # spektral kontrast bant sayısı (-> 7 değer)
CONTRAST_FMIN = 200.0             # spektral kontrast en düşük bant sınırı (Hz)
ROLLOFF_PERCENT = 0.85            # spektral rolloff yüzdesi
N_CHROMA = 12

# STFT tabanlı tüm özniteliklere ortak geçen parametreler
STFT_KW = dict(n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
               window=WINDOW, center=CENTER, pad_mode=PAD_MODE)


# ----------------------------------------------------------------------
# ÖN İŞLEME: sabit uzunluk
# ----------------------------------------------------------------------
def prepare_signal(y, sr=SAMPLE_RATE):
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


# ----------------------------------------------------------------------
# ÖZNİTELİK ÇIKARIM FONKSİYONU
# ----------------------------------------------------------------------
def extract_features_from_signal(y, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    """
    Bir sinyalden 1D öznitelik vektörü (sözlük) çıkarır.
    Her öznitelik zaman ekseninde ortalama (mean) ve standart sapma (std)
    olarak özetlenir, böylece değişken uzunluktaki sesler sabit uzunlukta
    bir vektöre dönüşür. Toplam: 51 öznitelik x 2 = 102 boyut.
    (6 skaler + 7 kontrast + 13 MFCC + 13 MFCC-delta + 12 chroma = 51)
    """
    y = prepare_signal(y, sr)
    features = {}

    # --- Zaman domeni öznitelikleri ---
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=N_FFT, hop_length=HOP_LENGTH, center=CENTER)
    features["zcr_mean"] = np.mean(zcr)
    features["zcr_std"] = np.std(zcr)

    rms = librosa.feature.rms(
        y=y, frame_length=N_FFT, hop_length=HOP_LENGTH, center=CENTER, pad_mode=PAD_MODE)
    features["rms_mean"] = np.mean(rms)
    features["rms_std"] = np.std(rms)

    # --- Spektral öznitelikler ---
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, **STFT_KW)
    features["spec_centroid_mean"] = np.mean(spec_centroid)
    features["spec_centroid_std"] = np.std(spec_centroid)

    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, **STFT_KW)
    features["spec_bandwidth_mean"] = np.mean(spec_bandwidth)
    features["spec_bandwidth_std"] = np.std(spec_bandwidth)

    spec_rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, roll_percent=ROLLOFF_PERCENT, **STFT_KW)
    features["spec_rolloff_mean"] = np.mean(spec_rolloff)
    features["spec_rolloff_std"] = np.std(spec_rolloff)

    spec_flatness = librosa.feature.spectral_flatness(y=y, **STFT_KW)
    features["spec_flatness_mean"] = np.mean(spec_flatness)
    features["spec_flatness_std"] = np.std(spec_flatness)

    spec_contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr, n_bands=CONTRAST_BANDS, fmin=CONTRAST_FMIN, **STFT_KW)
    for i in range(spec_contrast.shape[0]):
        features[f"spec_contrast_{i}_mean"] = np.mean(spec_contrast[i])
        features[f"spec_contrast_{i}_std"] = np.std(spec_contrast[i])

    # --- MFCC ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_mels=N_MELS, **STFT_KW)
    for i in range(mfcc.shape[0]):
        features[f"mfcc_{i}_mean"] = np.mean(mfcc[i])
        features[f"mfcc_{i}_std"] = np.std(mfcc[i])

    # MFCC delta (birinci türev) - zamansal değişimi yakalar, drone motor
    # gürültüsü gibi sürekli sesler için faydalı olabilir
    mfcc_delta = librosa.feature.delta(mfcc, width=DELTA_WIDTH, order=1, mode=DELTA_MODE)
    for i in range(mfcc_delta.shape[0]):
        features[f"mfcc_delta_{i}_mean"] = np.mean(mfcc_delta[i])
        features[f"mfcc_delta_{i}_std"] = np.std(mfcc_delta[i])

    # --- Chroma ---
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA, **STFT_KW)
    for i in range(chroma.shape[0]):
        features[f"chroma_{i}_mean"] = np.mean(chroma[i])
        features[f"chroma_{i}_std"] = np.std(chroma[i])

    # NOT: duration_sec bilinçli olarak EKLENMİYOR (kayıt uzunluğu artefaktı).
    return features


def extraction_parameters():
    """Makaleye yazılacak tüm çıkarım parametreleri (JSON'a kaydedilir)."""
    return {
        "sample_rate_hz": SAMPLE_RATE, "mono": True,
        "fixed_length_sec": FIXED_LENGTH_SEC, "crop_mode": CROP_MODE,
        "short_clip_padding": "wrap (döngüsel tekrar)" if FIXED_LENGTH_SEC else None,
        "n_fft": N_FFT, "win_length": WIN_LENGTH, "hop_length": HOP_LENGTH,
        "window": WINDOW, "center": CENTER, "pad_mode": PAD_MODE,
        "n_mfcc": N_MFCC, "n_mels": N_MELS,
        "delta_width": DELTA_WIDTH, "delta_mode": DELTA_MODE,
        "contrast_bands": CONTRAST_BANDS, "contrast_fmin_hz": CONTRAST_FMIN,
        "rolloff_percent": ROLLOFF_PERCENT, "n_chroma": N_CHROMA,
        "summary_statistics": ["mean", "std (ddof=0)"],
        "librosa_version": librosa.__version__, "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "machine": platform.platform(), "processor": platform.processor(),
    }


# ----------------------------------------------------------------------
# ANA AKIŞ: klasörleri gez, öznitelikleri çıkar, CSV'ye kaydet
# ----------------------------------------------------------------------
def main():
    rows = []
    t_load, t_feat = [], []

    for folder_name, label in CLASS_FOLDERS.items():
        folder_path = os.path.join(ROOT_DIR, folder_name)
        if not os.path.isdir(folder_path):
            print(f"UYARI: klasör bulunamadı, atlanıyor -> {folder_path}")
            continue

        file_list = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(AUDIO_EXTENSIONS)
        ]

        print(f"'{folder_name}' klasöründe {len(file_list)} ses dosyası bulundu.")

        for fname in tqdm(file_list, desc=f"{folder_name} işleniyor"):
            fpath = os.path.join(folder_path, fname)
            try:
                t0 = time.perf_counter()
                y, sr = librosa.load(fpath, sr=SAMPLE_RATE, mono=True)
                t1 = time.perf_counter()

                # Sessiz/boş dosya kontrolü
                if y is None or len(y) == 0:
                    print(f"  Atlandı (boş/okunamadı): {fname}")
                    continue

                feats = extract_features_from_signal(y, sr)
                t2 = time.perf_counter()

                t_load.append((t1 - t0) * 1000.0)
                t_feat.append((t2 - t1) * 1000.0)

                feats["filename"] = fname
                feats["label_name"] = folder_name
                feats["label"] = label
                rows.append(feats)
            except Exception as e:
                print(f"  HATA ({fname}): {e}")

    if not rows:
        print("Hiç öznitelik çıkarılamadı. ROOT_DIR ve klasör isimlerini kontrol edin.")
        return

    df = pd.DataFrame(rows)

    # filename, label_name, label sütunlarını başa al, geri kalanı sona
    meta_cols = ["filename", "label_name", "label"]
    other_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + other_cols]

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nTamamlandı. {len(df)} örnek, {len(other_cols)} öznitelik -> {OUTPUT_CSV}")
    print("\nSınıf dağılımı:")
    print(df["label_name"].value_counts())

    # --- Parametreler ve süre ölçümleri ---
    info = extraction_parameters()
    info["n_samples"] = int(len(df))
    info["feature_vector_size"] = int(len(other_cols))
    info["timing_ms_per_sample"] = {
        "load_resample_median": float(np.median(t_load)),
        "features_median": float(np.median(t_feat)),
        "features_mean": float(np.mean(t_feat)),
        "features_p95": float(np.percentile(t_feat, 95)),
        "load_plus_features_median": float(np.median(np.array(t_load) + np.array(t_feat))),
        "note": "Bu süreler betiğin çalıştığı bilgisayara aittir (ilk dosya JIT ısınmasını içerebilir).",
    }
    with open(OUTPUT_INFO_JSON, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"\nÖznitelik vektörü boyutu: {len(other_cols)}")
    print(f"Öznitelik çıkarım süresi (medyan): {info['timing_ms_per_sample']['features_median']:.2f} ms/örnek "
          f"(yükleme dahil: {info['timing_ms_per_sample']['load_plus_features_median']:.2f} ms)")
    print(f"Parametreler -> {OUTPUT_INFO_JSON}")


if __name__ == "__main__":
    main()
