"""
app/routers/disease_info.py
=================================================================
GET /disease-info/{disease_name}

Given a disease name (typically the `disease` value the app already
got back from /predict), returns:
  - clinical_info: primary/secondary symptoms, red flags, first aid
    (from the "suggest doctors" sheet of the doctor dataset)
  - doctors: the full doctor list for that disease's specialist

Coverage note: the doctor dataset currently only has rich clinical
detail for ~23 diseases (not all 494 the ML model can predict). When
a disease isn't covered, `found` is false and both `clinical_info`
and `doctors` come back empty — the Flutter app should hide this
section rather than show an error in that case.
"""

from fastapi import APIRouter

from app.data_loader import bundle
from app.schemas import (
    ClinicalInfoOut,
    DiseaseInfoResponse,
    DoctorListItem,
    DoctorOut,
    DoctorsBySpecialistResponse,
    DoctorsListResponse,
)

router = APIRouter(tags=["disease-info"])


@router.get("/doctors", response_model=DoctorsListResponse)
def get_all_doctors():
    """
    Full doctor directory across all specialties in the dataset (not just
    the ones currently linked to a disease) — powers the standalone
    "Find Doctors" screen with search-by-name and filter-by-specialty.
    """
    doctors: list[DoctorListItem] = []
    for specialty, doctor_list in bundle.doctors_directory.items():
        for d in doctor_list:
            doctors.append(DoctorListItem(specialty=specialty, **d))

    return DoctorsListResponse(
        specialties=sorted(bundle.doctors_directory.keys()),
        total=len(doctors),
        doctors=doctors,
    )


@router.get("/doctors/by-specialist", response_model=DoctorsBySpecialistResponse)
def get_doctors_by_specialist(specialist: str):
    """
    Resolve the ML model's `recommended_specialist` (e.g. "Dermatologist",
    or a combo like "Dermatologist / General Physician") to the matching
    doctor list. `found: false` means the dataset doesn't have doctors
    for this specialty yet — the Flutter app should show a friendly
    "coming soon" message in that case, not an error.
    """
    matched_specialty, doctor_list = bundle.lookup_doctors_by_specialist(specialist)
    return DoctorsBySpecialistResponse(
        found=matched_specialty is not None,
        requested_specialist=specialist,
        matched_specialty=matched_specialty,
        doctors=[DoctorOut(**d) for d in doctor_list],
    )


@router.get("/disease-info/{disease_name}", response_model=DiseaseInfoResponse)
def get_disease_info(disease_name: str):
    info = bundle.lookup_disease_info(disease_name)

    if info is None:
        return DiseaseInfoResponse(
            found=False,
            disease_name=disease_name,
            clinical_info=None,
            doctors=[],
        )

    specialist = info.get("specialist")
    doctors_raw = bundle.doctors_directory.get(specialist, []) if specialist else []

    return DiseaseInfoResponse(
        found=True,
        disease_name=disease_name,
        clinical_info=ClinicalInfoOut(**info),
        doctors=[DoctorOut(**d) for d in doctors_raw],
    )
