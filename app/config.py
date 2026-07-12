"""
app/config.py
=================================================================
Central place for paths and tunables. Nothing here should need to
change when you deploy — only DATA_DIR if you move the data folder.
"""

from pathlib import Path

# backend/app/config.py -> backend/
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

MODEL_PATH = DATA_DIR / "model.pkl"
LABEL_ENCODER_PATH = DATA_DIR / "label_encoder.pkl"
FEATURE_COLUMNS_PATH = DATA_DIR / "feature_columns.json"

DISEASE_DESCRIPTION_CSV = DATA_DIR / "disease_description.csv"
PRECAUTIONS_CSV = DATA_DIR / "precautions.csv"
SPECIALIST_CSV = DATA_DIR / "specialist.csv"
SYMPTOM_DICTIONARY_CSV = DATA_DIR / "symptom_dictionary.csv"
SYMPTOM_SEVERITY_CSV = DATA_DIR / "symptom_severity.csv"

CLINICAL_INFO_JSON = DATA_DIR / "clinical_info.json"
DOCTORS_DIRECTORY_JSON = DATA_DIR / "doctors_directory.json"

# Number of differential-diagnosis candidates to return
TOP_K = 3

# Severity level -> generic risk bucket used to derive the Flutter
# app's riskLevel ('Low' | 'Medium' | 'High' | 'Emergency')
SEVERITY_TO_RISK = {
    "Mild": "Low",
    "Moderate": "Medium",
    "Severe": "High",
}

# A selected symptom at/above this weight (out of the observed 1-7
# scale in symptom_severity.csv) additionally flags the whole result
# as a possible emergency, on top of the "Severe" bucket above.
EMERGENCY_WEIGHT_THRESHOLD = 6
