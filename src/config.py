"""Central path configuration for the PCVAE package.

All paths default to locations relative to the project root, so the code runs
from a plain checkout without edits. Point the code at external data / weights
by setting the environment variables below.

Environment variables
---------------------
PCVAE_ROOT           Root of the PCVAE project (default: the parent of ``src/``,
                     i.e. the directory that holds ``data/``, ``save/`` and
                     ``output/``).
PCVAE_DATA_DIR       Training / validation / test CSV directory.
PCVAE_SAVE_DIR       Model checkpoint directory.
PCVAE_OUTPUT_DIR     Property-predictor output directory.
PCVAE_GENERATED_DIR  Generated-sequence directory (kept as "genarated" so the
                     historical cross-references in this repo still resolve).
"""

import os
from pathlib import Path

# This file lives in <PCVAE_ROOT>/src/, so the project root is its parent.
SRC_DIR = Path(__file__).resolve().parent
PCVAE_ROOT = Path(os.environ.get("PCVAE_ROOT", SRC_DIR.parent))

DATA_DIR = Path(os.environ.get("PCVAE_DATA_DIR", PCVAE_ROOT / "data"))
SAVE_DIR = Path(os.environ.get("PCVAE_SAVE_DIR", PCVAE_ROOT / "save"))
OUTPUT_DIR = Path(os.environ.get("PCVAE_OUTPUT_DIR", PCVAE_ROOT / "output"))
GENERATED_DIR = Path(os.environ.get("PCVAE_GENERATED_DIR", PCVAE_ROOT / "genarated"))

# Prediction outputs produced by the models under PCVAE/classifier/.
CLASSIFIER_OUTPUT_DIR = OUTPUT_DIR / "classifier"
