"""
app/ml_service.py
=================================================================
Core prediction logic: builds the 374-length feature vector in the
exact order of feature_columns.json, runs model.predict_proba, maps
back to disease names via label_encoder, and joins in description /
precautions / specialist from the static lookup tables.

Also derives a Flutter-friendly riskLevel / isEmergency / advice
bundle from symptom_severity.csv, since the model itself only
outputs a disease + confidence, not a risk level.
"""

import numpy as np

from app import config
from app.data_loader import bundle


class EmptySymptomsError(ValueError):
    """Raised when the caller provides no usable symptoms at all."""


def normalize_symptom(name: str) -> str:
    """Trim whitespace; matching is otherwise exact per the project's
    guarantee that names line up 1:1 across all files."""
    return name.strip()


def split_known_unknown(symptoms: list[str]) -> tuple[list[str], list[str]]:
    """Split the incoming symptom list into (known, ignored) based on
    whether each name exists in feature_columns.json."""
    known, ignored = [], []
    seen = set()
    for raw in symptoms:
        name = normalize_symptom(raw)
        if not name or name in seen:
            continue
        seen.add(name)
        if name in bundle.feature_index:
            known.append(name)
        else:
            ignored.append(raw)
    return known, ignored


def build_feature_vector(known_symptoms: list[str]) -> np.ndarray:
    vector = np.zeros((1, len(bundle.feature_columns)), dtype=float)
    for name in known_symptoms:
        vector[0, bundle.feature_index[name]] = 1.0
    return vector


def top_k_predictions(vector: np.ndarray, k: int = config.TOP_K) -> list[dict]:
    proba = bundle.model.predict_proba(vector)[0]
    top_idx = np.argsort(proba)[::-1][:k]

    results = []
    for idx in top_idx:
        disease = bundle.label_encoder.classes_[idx]
        confidence = float(proba[idx])
        results.append(
            {
                "disease": disease,
                "confidence": round(confidence, 4),
                "description": bundle.disease_description.get(disease, ""),
                "precautions": bundle.precautions.get(disease, []),
                "recommended_specialist": bundle.specialist.get(disease, "General Physician"),
            }
        )
    return results


def assess_risk(known_symptoms: list[str]) -> dict:
    """
    Derive a Flutter-facing risk bundle purely from the severity of the
    symptoms the user selected (independent of the model's confidence).
    """
    levels_seen = set()
    max_weight = 0
    emergency_symptom = None

    for name in known_symptoms:
        info = bundle.symptom_severity.get(name)
        if not info:
            continue
        levels_seen.add(info["level"])
        if info["weight"] > max_weight:
            max_weight = info["weight"]
        if info["weight"] >= config.EMERGENCY_WEIGHT_THRESHOLD and emergency_symptom is None:
            emergency_symptom = name

    if emergency_symptom:
        risk_level = "Emergency"
        is_emergency = True
        emergency_warning = (
            f"⚠️ '{emergency_symptom}' can indicate a serious condition. "
            "If you are in distress, seek emergency medical care immediately."
        )
    elif "Severe" in levels_seen:
        risk_level = config.SEVERITY_TO_RISK["Severe"]
        is_emergency = False
        emergency_warning = None
    elif "Moderate" in levels_seen:
        risk_level = config.SEVERITY_TO_RISK["Moderate"]
        is_emergency = False
        emergency_warning = None
    else:
        risk_level = config.SEVERITY_TO_RISK["Mild"]
        is_emergency = False
        emergency_warning = None

    return {
        "riskLevel": risk_level,
        "isEmergency": is_emergency,
        "emergencyWarning": emergency_warning,
    }


def predict(symptoms: list[str]) -> dict:
    known, ignored = split_known_unknown(symptoms)

    if not known:
        raise EmptySymptomsError(
            "No valid symptoms were provided. Please select at least one "
            "symptom from the /symptoms list."
        )

    vector = build_feature_vector(known)
    predictions = top_k_predictions(vector)
    risk = assess_risk(known)

    top = predictions[0]

    return {
        # Flutter PredictionResult-compatible top-level fields
        "disease": top["disease"],
        "confidence": top["confidence"],
        "riskLevel": risk["riskLevel"],
        "advice": top["description"],
        "specialist": top["recommended_specialist"],
        "symptoms": known,
        "isEmergency": risk["isEmergency"],
        "emergencyWarning": risk["emergencyWarning"],
        "generalAdvice": top["precautions"],
        "warningNote": (
            "This is an AI-generated preliminary assessment, not a medical "
            "diagnosis. Please consult a qualified healthcare provider."
        ),
        # Extra fields: full differential + validation feedback
        "predictions": predictions,
        "ignored_symptoms": ignored,
    }
