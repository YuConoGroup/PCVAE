"""Central path configuration for the toxicity classifier.

Defaults are relative to this file so the module runs from a plain checkout.
Override with environment variables to use external data / pretrained models.

Environment variables
---------------------
TOX_CLASSIFIER_DIR  Module root (default: this directory).
TOX_ESM2_DIR        Local ESM-2 t12_35M model directory.
TOX_DATA_DIR        Input CSV directory.
TOX_FEATURES_DIR    PCA-reduced feature directory.
TOX_RESULT_DIR      Inference output directory.
"""

import os
from pathlib import Path

BASE_DIR = Path(os.environ.get("TOX_CLASSIFIER_DIR", Path(__file__).resolve().parent))

ESM2_DIR = Path(os.environ.get("TOX_ESM2_DIR", BASE_DIR / "esm2_t12_35M_UR50D"))
DATA_DIR = Path(os.environ.get("TOX_DATA_DIR", BASE_DIR / "data"))
FEATURES_DIR = Path(os.environ.get("TOX_FEATURES_DIR", BASE_DIR / "features"))
RESULT_DIR = Path(os.environ.get("TOX_RESULT_DIR", BASE_DIR / "generated"))

SAVE_DIR = BASE_DIR / "save"
MODEL_DIR = SAVE_DIR / "save_model"
PICTURE_DIR = SAVE_DIR / "picture"

# Cached artefacts produced by extract_features.py / train.py.
FEATURES_PKL = FEATURES_DIR / "features.pkl"
PCA_PKL = FEATURES_DIR / "pca_model.pkl"
SCALER_PKL = Path(os.environ.get("TOX_SCALER_PKL", BASE_DIR / "scaler.pkl"))
