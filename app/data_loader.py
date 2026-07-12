"""
app/data_loader.py
=================================================================
Loads the trained model bundle + static lookup CSVs exactly once,
at process startup, and exposes them as module-level singletons.

Nothing here retrains or modifies model.pkl / label_encoder.pkl /
feature_columns.json — they are loaded as-is per the project
constraint.
"""

import json
import logging
import re

import joblib
import pandas as pd

from app import config

logger = logging.getLogger("smart_health_assistant.data_loader")

# The doctor/clinical dataset ("suggest doctors" sheet) uses slightly
# different disease names than the ML model's label_encoder classes for
# a handful of entries. Map: dataset name -> exact model class name.
_DISEASE_NAME_TO_MODEL = {
    "Kidney Stones": "Kidney Stone",
    "Diabetes Mellitus (Type 2)": "Diabetes (Type 2)",
    "Peptic Ulcer": "Peptic Ulcer Disease",
    "IBS": "Irritable Bowel Syndrome (IBS)",
}


# A recommended-specialist string from specialist.csv can use different
# wording than the doctor dataset's specialty keys, or be a "/"-separated
# list of candidates (e.g. "Dermatologist / General Physician"). Map the
# wording differences here; combo-splitting is handled in the lookup.
_SPECIALIST_ALIASES = {
    "infectious disease specialist": "ID Specialist",
    "allergist": "Allergists & Immunologists",
    "immunologist": "Allergists & Immunologists",
}


def _normalize(name: str) -> str:
    """Loose key for fallback matching: lowercase, drop parenthetical
    text and punctuation. 'GERD (Acid Reflux)' -> 'gerd'."""
    name = re.sub(r"\([^)]*\)", "", name)
    name = re.sub(r"[^a-z0-9]+", "", name.lower())
    return name


class DataBundle:
    def __init__(self):
        logger.info("Loading model bundle...")
        self.model = joblib.load(config.MODEL_PATH)
        self.label_encoder = joblib.load(config.LABEL_ENCODER_PATH)

        with open(config.FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as fh:
            self.feature_columns: list[str] = json.load(fh)

        # Fast O(1) lookup: symptom name -> index in the feature vector
        self.feature_index = {name: i for i, name in enumerate(self.feature_columns)}

        logger.info(
            "Model loaded: %s | %d features | %d disease classes",
            type(self.model).__name__,
            len(self.feature_columns),
            len(self.label_encoder.classes_),
        )

        # ---- Static lookup/reference tables ----
        self.disease_description = self._load_disease_description()
        self.precautions = self._load_precautions()
        self.specialist = self._load_specialist()
        self.symptom_dictionary = self._load_symptom_dictionary()
        self.symptom_severity = self._load_symptom_severity()

        # ---- Doctor directory / rich clinical info (optional dataset) ----
        self.clinical_info = self._load_clinical_info()
        self.doctors_directory = self._load_doctors_directory()
        self._clinical_info_normalized_index = {
            _normalize(name): name for name in self.clinical_info
        }

        self._sanity_check()

    # -- loaders ------------------------------------------------------

    def _load_disease_description(self) -> dict:
        df = pd.read_csv(config.DISEASE_DESCRIPTION_CSV)
        return dict(zip(df["Disease"], df["Description"]))

    def _load_precautions(self) -> dict:
        df = pd.read_csv(config.PRECAUTIONS_CSV)
        precaution_cols = [c for c in df.columns if c.startswith("Precaution_")]
        result = {}
        for _, row in df.iterrows():
            items = [
                str(row[c]).strip()
                for c in precaution_cols
                if pd.notna(row[c]) and str(row[c]).strip()
            ]
            result[row["Disease"]] = items
        return result

    def _load_specialist(self) -> dict:
        df = pd.read_csv(config.SPECIALIST_CSV)
        return dict(zip(df["Disease"], df["Recommended_Specialist"]))

    def _load_symptom_dictionary(self) -> pd.DataFrame:
        return pd.read_csv(config.SYMPTOM_DICTIONARY_CSV)

    def _load_symptom_severity(self) -> dict:
        """Symptom name -> {"weight": int, "level": str}"""
        df = pd.read_csv(config.SYMPTOM_SEVERITY_CSV)
        result = {}
        for _, row in df.iterrows():
            result[row["Symptom"]] = {
                "weight": int(row["Severity_Weight"]),
                "level": row["Severity_Level"],
            }
        return result

    def _load_clinical_info(self) -> dict:
        """disease name (as written in the doctor dataset) -> clinical detail dict"""
        if not config.CLINICAL_INFO_JSON.exists():
            logger.warning("clinical_info.json not found; /disease-info will be empty.")
            return {}
        with open(config.CLINICAL_INFO_JSON, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _load_doctors_directory(self) -> dict:
        """specialist name -> list of doctor dicts"""
        if not config.DOCTORS_DIRECTORY_JSON.exists():
            logger.warning("doctors_directory.json not found; /disease-info will be empty.")
            return {}
        with open(config.DOCTORS_DIRECTORY_JSON, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def lookup_disease_info(self, disease_name: str) -> dict | None:
        """
        Resolve a disease name (typically the ML model's predicted class
        name) to its clinical_info + doctors_directory entry, if the
        doctor dataset covers that disease. Tries, in order:
          1. exact match against clinical_info keys
          2. reverse alias match (model name -> dataset name)
          3. normalized match (case/punctuation/parenthetical-insensitive)
        Returns None if the doctor dataset simply doesn't cover this
        disease yet (expected for most of the model's 494 classes —
        the dataset only has rich info for ~23 diseases so far).
        """
        if disease_name in self.clinical_info:
            return self.clinical_info[disease_name]

        for dataset_name, model_name in _DISEASE_NAME_TO_MODEL.items():
            if model_name == disease_name and dataset_name in self.clinical_info:
                return self.clinical_info[dataset_name]

        key = _normalize(disease_name)
        matched_name = self._clinical_info_normalized_index.get(key)
        if matched_name:
            return self.clinical_info[matched_name]

        return None

    def lookup_doctors_by_specialist(self, specialist_name: str):
        """
        Resolve a `recommended_specialist` string (from specialist.csv —
        can be a single name or a "/"-separated list of candidates like
        "Dermatologist / General Physician") to a doctors_directory
        entry. Returns (matched_specialty_name, doctor_list); matched
        name is None and the list is empty when the dataset doesn't
        cover any of the candidate specialties yet.
        """
        candidates = [c.strip() for c in specialist_name.split("/") if c.strip()]
        directory_keys_lower = {k.lower(): k for k in self.doctors_directory}

        for candidate in candidates:
            key_lower = candidate.lower()
            if key_lower in directory_keys_lower:
                matched = directory_keys_lower[key_lower]
                return matched, self.doctors_directory[matched]
            alias = _SPECIALIST_ALIASES.get(key_lower)
            if alias and alias in self.doctors_directory:
                return alias, self.doctors_directory[alias]

        return None, []

    # -- integrity check ------------------------------------------------

    def _sanity_check(self):
        """
        Warn (don't crash) if the lookup tables are missing anything the
        model can predict / accept. The project README states this was
        already verified to be 0-missing, but we check defensively in
        case the CSVs are swapped out later.
        """
        missing_disease_desc = [
            d for d in self.label_encoder.classes_ if d not in self.disease_description
        ]
        missing_specialist = [
            d for d in self.label_encoder.classes_ if d not in self.specialist
        ]
        missing_precautions = [
            d for d in self.label_encoder.classes_ if d not in self.precautions
        ]
        missing_severity = [
            s for s in self.feature_columns if s not in self.symptom_severity
        ]

        for name, missing in [
            ("disease_description.csv", missing_disease_desc),
            ("specialist.csv", missing_specialist),
            ("precautions.csv", missing_precautions),
            ("symptom_severity.csv", missing_severity),
        ]:
            if missing:
                logger.warning(
                    "%s is missing %d entr(y/ies) referenced by the model: %s",
                    name,
                    len(missing),
                    missing[:10],
                )


# Singleton, imported by routers/services
bundle = DataBundle()
