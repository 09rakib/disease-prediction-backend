"""
app/schemas.py
=================================================================
Pydantic models for request/response validation.

`PredictResponse` is shaped to be a drop-in match for the Flutter
app's `PredictionResult.fromJson` (lib/models/prediction_result.dart)
— same key names (disease, confidence, riskLevel, advice, specialist,
symptoms, isEmergency, emergencyWarning, generalAdvice, warningNote)
— plus two extra fields (`predictions`, `ignored_symptoms`) that the
current Flutter model simply ignores but are there for the full
Top-3 differential + validation feedback.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class SymptomOut(BaseModel):
    id: str
    name: str
    severity_level: Optional[str] = None
    severity_weight: Optional[int] = None


class SymptomsResponse(BaseModel):
    count: int
    symptoms: List[SymptomOut]


class PredictRequest(BaseModel):
    symptoms: List[str] = Field(..., description="List of user-selected symptom names")


class DiseasePrediction(BaseModel):
    disease: str
    confidence: float
    description: str
    precautions: List[str]
    recommended_specialist: str


class PredictResponse(BaseModel):
    # ---- Flutter PredictionResult-compatible fields (top match) ----
    disease: str
    confidence: float
    riskLevel: str
    advice: str
    specialist: str
    symptoms: List[str]
    isEmergency: bool
    emergencyWarning: Optional[str] = None
    generalAdvice: List[str]
    warningNote: str

    # ---- Extra fields (full Top-3 differential + diagnostics) ----
    predictions: List[DiseasePrediction]
    ignored_symptoms: List[str]


class DoctorOut(BaseModel):
    name: str
    qualifications: Optional[str] = None
    position: Optional[str] = None
    focus_areas: Optional[str] = None
    chamber_address: Optional[str] = None
    hours: Optional[str] = None
    contact_phones: List[str] = Field(default_factory=list)


class ClinicalInfoOut(BaseModel):
    department: Optional[str] = None
    specialist: Optional[str] = None
    disease_name: str
    primary_symptoms: Optional[str] = None
    secondary_symptoms: Optional[str] = None
    red_flags: Optional[str] = None
    first_aid: Optional[str] = None


class DiseaseInfoResponse(BaseModel):
    found: bool
    disease_name: str
    clinical_info: Optional[ClinicalInfoOut] = None
    doctors: List[DoctorOut] = Field(default_factory=list)


class DoctorListItem(DoctorOut):
    specialty: str


class DoctorsListResponse(BaseModel):
    specialties: List[str]
    total: int
    doctors: List[DoctorListItem]


class DoctorsBySpecialistResponse(BaseModel):
    found: bool
    requested_specialist: str
    matched_specialty: Optional[str] = None
    doctors: List[DoctorOut] = Field(default_factory=list)
