"""
app/main.py
=================================================================
FastAPI entrypoint for the Smart Health Assistant backend.

Run with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.data_loader import bundle  # noqa: F401  (triggers model load at import time)
from app.routers import disease_info, predict, symptoms
from app.history import router as history_router, init_history_db

logging.basicConfig(level=logging.INFO)

# Creates data/history.db (and the predictions table) if it doesn't exist yet.
init_history_db()

app = FastAPI(
    title="Smart Health Assistant API",
    description=(
        "Symptom-based disease prediction API. Serves a pre-trained "
        "scikit-learn model (374 symptoms -> 494 disease classes) plus "
        "static lookup data (descriptions, precautions, specialists)."
    ),
    version="1.0.0",
)

# Flutter mobile app calls this from a device/emulator, not a browser
# page under the same origin, so allow all origins (fine for a
# dev/thesis project; tighten allow_origins for production).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(symptoms.router)
app.include_router(predict.router)
app.include_router(disease_info.router)
app.include_router(history_router)


@app.get("/", tags=["health"])
def root():
    return {
        "status": "ok",
        "service": "Smart Health Assistant API",
        "model_classes": len(bundle.label_encoder.classes_),
        "feature_count": len(bundle.feature_columns),
    }


@app.get("/health", tags=["health"])
def health():
    return {"status": "healthy"}