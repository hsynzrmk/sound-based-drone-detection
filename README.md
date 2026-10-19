# Sound-Based Drone Detection Using Machine Learning

Code for the paper **"Sound-Based Drone Detection Using Machine Learning Methods"** (Harran University Journal of Engineering).

The repository contains the feature extraction and the training/evaluation pipeline used to compare 14 classical machine learning algorithms on compact acoustic features for two tasks:

- **Drone recognition (binary):** drone vs. non-drone (`yes_drone` / `unknown`)
- **Drone classification (three-class):** `bebop` / `membo` / `unknown`

## Repository contents

| File | Description |
|---|---|
| `detect_all_models.py` | Feature extraction, training, evaluation and efficiency benchmarking |
| `requirements.txt` | Python dependencies (pin your own versions) |
| `features.csv` | Extracted feature table (optional, if shared) |
| `results/` | Output tables and figures written by the script |

## Installation

```bash
git clone https://github.com/<USERNAME>/<REPOSITORY>.git
cd <REPOSITORY>
pip install -r requirements.txt
```

Main dependencies: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `matplotlib`, `librosa`, `soundfile`, `psutil`.
If `xgboost` or `lightgbm` is not installed, the script skips those models and continues with the rest.

## Datasets

The audio files are **not** redistributed in this repository. Please download them from the original sources and follow their licenses.

| Dataset | Use in the paper | Link |
|---|---|---|
| Drone Audio Dataset (Al-Emadi et al.) | Main experiments (binary and three-class) | https://github.com/saraalemadi/DroneAudioDataset |
| ESC-50 | Source of non-drone (`unknown`) clips inside the Drone Audio Dataset | https://github.com/karoldvl/ESC-50 |
| Drone Detection Dataset | Additional three-class experiment (drone / helicopter / background, 90 recordings) | https://github.com/DroneDetectionThesis/Drone-detection-dataset |

## Feature extraction

All recordings are converted to mono, resampled to 22,050 Hz and fixed to 1 s. Features are computed with `librosa` using the following fixed parameters:

| Parameter | Value |
|---|---|
| Sampling rate | 22,050 Hz (mono) |
| `n_fft` / `win_length` | 2048 / 2048 |
| `hop_length` | 512 |
| Window | Hann |
| `n_mfcc` | 13 |
| `n_chroma` | 12 |
| Spectral contrast bands | 6 (7 values) |

51 frame-level features are summarized by their **mean and standard deviation**, giving a **102-dimensional** vector per clip:

ZCR (1), RMS (1), spectral centroid (1), bandwidth (1), rolloff (1), flatness (1), spectral contrast (7), MFCC (13), MFCC delta (13), chroma (12).

The clip duration (`duration_sec`) is not used as a model input, because it is a dataset artifact rather than an acoustic property.

## Usage

### 1. Extract features from a folder of audio files

```bash
python detect_all_models.py --extract <AUDIO_FOLDER> <LABEL_NAME> <OUTPUT_CSV>
```

Example:

```bash
python detect_all_models.py --extract ./audio/bebop  bebop   bebop.csv
python detect_all_models.py --extract ./audio/other  unknown unknown.csv
```

Merge the resulting CSV files into a single `features.csv` with the columns `filename, label_name, label, <102 feature columns>`.

### 2. Train and evaluate all models

```bash
# Three-class task
python detect_all_models.py --csv features.csv --task multiclass

# Binary task
python detect_all_models.py --csv features.csv --task binary

# Several random splits (mean ± std)
python detect_all_models.py --csv features.csv --task multiclass --seeds 42 43 44 45 46

# Also measure feature-extraction time (folder with the wav files)
python detect_all_models.py --csv features.csv --task multiclass --audio_dir ./audio

# Train on the main data, test on an external feature CSV (same schema)
python detect_all_models.py --csv features.csv --task multiclass --external_csv external.csv
```

### Command-line arguments

| Argument | Default | Description |
|---|---|---|
| `--csv` | `features.csv` | Feature table |
| `--task` | `multiclass` | `binary` or `multiclass` |
| `--seeds` | `42` | One or more random seeds |
| `--audio_dir` | – | Audio folder for feature-extraction timing |
| `--external_csv` | – | External feature CSV for train-on-main / test-on-external evaluation |
| `--out_dir` | `results` | Output folder |
| `--extract` | – | `FOLDER LABEL OUTPUT_CSV`: extract features and exit |

### Outputs (`results/`)

| File | Content |
|---|---|
| `model_hyperparameters_<task>.csv` | Hyperparameters and loss/optimization criterion of every model |
| `model_comparison_<task>.csv` | Test and cross-validation metrics of every model |
| `confusion_matrices_<task>.png` | Confusion matrices |
| `efficiency_<task>.csv` | Model size, inference latency (median, p95), RAM increase, training time |
| `external_<task>.csv` | External-dataset results (if `--external_csv` is given) |

## Evaluation protocol

- **Record-wise split:** all clips derived from the same original recording are placed entirely in either the training or the test set (`StratifiedGroupKFold`). Record overlap is verified in the code.
- **Hold-out test:** 80 % / 20 %, class distribution preserved.
- **Cross-validation:** 5-fold grouped CV, run on the training set only. The test set is used only for the final evaluation.
- **Metrics:** accuracy, macro precision/recall, macro and weighted F1, balanced accuracy, MCC, and a majority-class baseline.
- **Reproducibility:** `random_state = 42` for all models and splits.
- **Efficiency:** inference latency is measured with batch size 1 on a single thread; RAM increase is approximate.

## Models

Logistic Regression, SVM (RBF), SVM (Linear), K-Nearest Neighbors, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, Gaussian Naive Bayes, Linear Discriminant Analysis, MLP, XGBoost, LightGBM.
Full hyperparameters are written to `results/model_hyperparameters_<task>.csv`.

## Main results (from the paper)

| Task | Best model | Test accuracy | Macro F1 | CV accuracy |
|---|---|---|---|---|
| Drone recognition (Drone Audio Dataset) | LightGBM | 97.49 % | 0.9356 | 98.13 % |
| Three-class classification (Drone Audio Dataset) | LightGBM | 97.45 % | 0.8981 | 97.59 % |
| Three-class (Drone Detection Dataset, 90 recordings) | LightGBM | 88.89 % | 0.8889 | 79.43 % |

Efficiency (three-class task): boosting models are below 2 MB with median inference latency below 1 ms; Random Forest and Extra Trees are considerably larger (about 12 MB and 35 MB) and slower (about 11 ms).
Energy consumption was not measured, and the timings were obtained on a single desktop computer.

## Limitations

- The main dataset was largely recorded indoors and augmented with noise clips, so it does not fully represent outdoor effects such as wind noise, distance attenuation, Doppler shift and reverberation.
- Each clip carries a single label; simultaneous multiple drones are not modeled.
- The additional dataset contains only 90 recordings.

## Citation

```bibtex
@article{ezirmik_drone_audio,
  author  = {Ezirmik, Abdurrahim H{\"u}seyin},
  title   = {Sound-Based Drone Detection Using Machine Learning Methods},
  journal = {Harran University Journal of Engineering},
  year    = {2026}
}
```

*(Update the entry with volume, pages and DOI after publication.)*

## License

Released under the MIT License (see `LICENSE`). The datasets are subject to their own licenses.
