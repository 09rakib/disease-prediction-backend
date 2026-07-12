"""
history.py — Prediction History storage (device_id based, no login required)
================================================================================
Self-contained module. Add to your existing FastAPI app with just 2 lines
in main.py (see bottom of this file for exact instructions).

Storage: SQLite file at data/history.db (created automatically on first run).
No new pip packages needed — uses Python's built-in sqlite3.
"""

import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app import config

DB_PATH = config.DATA_DIR / "history.db"

router = APIRouter(prefix="/predictions", tags=["history"])


# ──────────────────────────────────────────────────────────────────────────
# DB setup
# ──────────────────────────────────────────────────────────────────────────
def init_history_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id                     TEXT PRIMARY KEY,
            device_id              TEXT NOT NULL,
            date_time              TEXT NOT NULL,
            disease_name           TEXT NOT NULL,
            confidence_score       REAL NOT NULL,
            risk_level             TEXT NOT NULL,
            doctor_name            TEXT NOT NULL,
            status                 TEXT NOT NULL DEFAULT 'completed',
            symptoms               TEXT NOT NULL,   -- comma-separated
            notes                  TEXT NOT NULL DEFAULT ''
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_predictions_device_id ON predictions(device_id)"
    )
    conn.commit()
    conn.close()


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _risk_level(confidence: float) -> str:
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.45:
        return "medium"
    return "low"


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "date_time": row["date_time"],
        "disease_name": row["disease_name"],
        "confidence_score": row["confidence_score"],
        "risk_level": row["risk_level"],
        "doctor_name": row["doctor_name"],
        "status": row["status"],
        "symptoms": row["symptoms"].split(",") if row["symptoms"] else [],
        "notes": row["notes"],
    }


# ──────────────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────────────
class PredictionRecordCreate(BaseModel):
    device_id: str = Field(..., description="Anonymous per-device identifier from the app")
    disease: str
    confidence: float
    recommended_specialist: str
    symptoms: list[str]
    notes: str = ""


class PredictionRecordOut(BaseModel):
    id: str
    date_time: str
    disease_name: str
    confidence_score: float
    risk_level: str
    doctor_name: str
    status: str
    symptoms: list[str]
    notes: str


class PredictionListOut(BaseModel):
    count: int
    records: list[PredictionRecordOut]


# ──────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────
@router.post("", response_model=PredictionRecordOut)
def save_prediction(payload: PredictionRecordCreate):
    if not payload.device_id.strip():
        raise HTTPException(status_code=400, detail="device_id is required.")
    if not payload.symptoms:
        raise HTTPException(status_code=400, detail="symptoms list cannot be empty.")

    record_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    risk = _risk_level(payload.confidence)

    conn = _get_conn()
    conn.execute(
        """
        INSERT INTO predictions
            (id, device_id, date_time, disease_name, confidence_score,
             risk_level, doctor_name, status, symptoms, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'completed', ?, ?)
        """,
        (
            record_id,
            payload.device_id,
            now,
            payload.disease,
            payload.confidence,
            risk,
            payload.recommended_specialist,
            ",".join(payload.symptoms),
            payload.notes,
        ),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM predictions WHERE id = ?", (record_id,)).fetchone()
    conn.close()
    return _row_to_dict(row)


@router.get("", response_model=PredictionListOut)
def list_predictions(device_id: str = Query(..., description="Device identifier to filter by")):
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM predictions WHERE device_id = ? ORDER BY date_time DESC",
        (device_id,),
    ).fetchall()
    conn.close()
    return {"count": len(rows), "records": [_row_to_dict(r) for r in rows]}


@router.get("/{record_id}", response_model=PredictionRecordOut)
def get_prediction(record_id: str):
    conn = _get_conn()
    row = conn.execute("SELECT * FROM predictions WHERE id = ?", (record_id,)).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Prediction record not found.")
    return _row_to_dict(row)


@router.delete("/{record_id}")
def delete_prediction(record_id: str, device_id: str = Query(...)):
    conn = _get_conn()
    row = conn.execute("SELECT * FROM predictions WHERE id = ?", (record_id,)).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail="Prediction record not found.")
    if row["device_id"] != device_id:
        conn.close()
        raise HTTPException(status_code=403, detail="This record belongs to a different device.")
    conn.execute("DELETE FROM predictions WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()
    return {"deleted": True, "id": record_id}


# ──────────────────────────────────────────────────────────────────────────
# HOW TO PLUG THIS INTO YOUR EXISTING main.py
# ──────────────────────────────────────────────────────────────────────────
# 1. Save this file as `history.py` in the SAME folder as your main.py
#    (so: backend/app/history.py if main.py is in backend/app/)
#
# 2. In main.py, add near the top with your other imports:
#       from history import router as history_router, init_history_db
#
# 3. Find your @app.on_event("startup") function and add one line:
#       @app.on_event("startup")
#       def on_startup():
#           load_resources()      # <- your existing line
#           init_history_db()     # <- ADD THIS LINE
#
# 4. Right after you create `app = FastAPI(...)`, add:
#       app.include_router(history_router)
#
# That's it — no changes to your existing /predict or /symptoms logic needed.
# New endpoints will be available at:
#   POST   /predictions
#   GET    /predictions?device_id=xxx
#   GET    /predictions/{id}
#   DELETE /predictions/{id}?device_id=xxx