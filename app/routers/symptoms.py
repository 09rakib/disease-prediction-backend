"""
app/routers/symptoms.py
=================================================================
GET /symptoms
"""

from fastapi import APIRouter

from app.data_loader import bundle
from app.schemas import SymptomOut, SymptomsResponse

router = APIRouter(tags=["symptoms"])


@router.get("/symptoms", response_model=SymptomsResponse)
def get_symptoms():
    """
    Returns all 374 symptoms known to the model, each annotated with its
    severity level/weight from symptom_severity.csv (for UI color-coding).
    """
    out = []
    for _, row in bundle.symptom_dictionary.iterrows():
        name = row["Symptom_Name"]
        sev = bundle.symptom_severity.get(name, {})
        out.append(
            SymptomOut(
                id=f"symptom_{int(row['Index'])}",
                name=name,
                severity_level=sev.get("level"),
                severity_weight=sev.get("weight"),
            )
        )
    return SymptomsResponse(count=len(out), symptoms=out)
