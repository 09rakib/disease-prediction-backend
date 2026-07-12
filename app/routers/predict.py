"""
app/routers/predict.py
=================================================================
POST /predict
"""

from fastapi import APIRouter, HTTPException

from app import ml_service
from app.schemas import PredictRequest, PredictResponse

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    if not payload.symptoms:
        raise HTTPException(
            status_code=400,
            detail="Please provide at least one symptom in the 'symptoms' list.",
        )

    try:
        result = ml_service.predict(payload.symptoms)
    except ml_service.EmptySymptomsError as exc:
        # All provided symptoms were unrecognized -> nothing to predict on.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return result
