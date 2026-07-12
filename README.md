# Smart Health Assistant — Backend API

FastAPI backend that serves the pre-trained disease-prediction model
(`model.pkl`, 374 symptoms → 494 disease classes, Top-1 ≈ 78%, Top-3 ≈ 91%)
to the Flutter app. Nothing here retrains the model — it only loads and
serves what was already trained.

## 1. Project structure

```
backend/
├── requirements.txt
├── data/                          # <-- model bundle + lookup CSVs live here
│   ├── model.pkl
│   ├── label_encoder.pkl
│   ├── feature_columns.json
│   ├── disease_description.csv
│   ├── precautions.csv
│   ├── specialist.csv
│   ├── symptom_dictionary.csv
│   └── symptom_severity.csv
└── app/
    ├── main.py                    # FastAPI app, CORS, router registration
    ├── config.py                  # paths + tunables (severity thresholds etc.)
    ├── schemas.py                 # Pydantic request/response models
    ├── data_loader.py             # loads model + CSVs once at startup
    ├── ml_service.py              # feature-vector building, predict, risk logic
    └── routers/
        ├── symptoms.py            # GET /symptoms
        └── predict.py             # POST /predict
```

## 2. Setup & run

```bash
cd backend
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000/docs` for interactive Swagger docs.

`scikit-learn==1.9.0` is pinned in `requirements.txt` because that's the
version `model.pkl` / `label_encoder.pkl` were saved with — using an older
sklearn will still work but prints an `InconsistentVersionWarning`.

## 3. Endpoints

### `GET /symptoms`
Returns all 374 symptoms the model understands, each tagged with severity
(from `symptom_severity.csv`) for UI color-coding.

```json
{
  "count": 374,
  "symptoms": [
    { "id": "symptom_0", "name": "Abdominal Distention", "severity_level": "Mild", "severity_weight": 3 },
    ...
  ]
}
```

### `POST /predict`
Request:
```json
{ "symptoms": ["Fever", "Cough", "Headache"] }
```

Response — shaped to match the Flutter app's `PredictionResult.fromJson`
one-to-one (`disease`, `confidence`, `riskLevel`, `advice`, `specialist`,
`symptoms`, `isEmergency`, `emergencyWarning`, `generalAdvice`,
`warningNote`), plus two extra fields the current Flutter model just
ignores but are useful (`predictions` = full Top-3 differential,
`ignored_symptoms` = any names that weren't recognized):

```json
{
  "disease": "Aphthous Ulcer",
  "confidence": 0.16,
  "riskLevel": "Medium",
  "advice": "Aphthous Ulcer is a medical condition requiring...",
  "specialist": "General Physician",
  "symptoms": ["Fever", "Cough", "Headache"],
  "isEmergency": false,
  "emergencyWarning": null,
  "generalAdvice": ["Consult a healthcare provider promptly", "..."],
  "warningNote": "This is an AI-generated preliminary assessment...",
  "predictions": [
    { "disease": "Aphthous Ulcer", "confidence": 0.16, "description": "...", "precautions": ["..."], "recommended_specialist": "General Physician" },
    { "disease": "Meningitis", "confidence": 0.1167, "...": "..." },
    { "disease": "Abscess Of The Pharynx", "confidence": 0.0668, "...": "..." }
  ],
  "ignored_symptoms": []
}
```

`riskLevel` / `isEmergency` are derived from the **severity of the
symptoms the user selected** (`symptom_severity.csv`), not from the
model's confidence — the model only predicts a disease, not a risk tier.
Any symptom with severity weight ≥ 6 (e.g. "Yellow Eyes") flags
`isEmergency: true` and fills in `emergencyWarning`. Tune the threshold
in `app/config.py` (`EMERGENCY_WEIGHT_THRESHOLD`).

**Error handling**
- Empty `symptoms` list → `400` with a clear message.
- Unknown symptom names → silently ignored, listed back in
  `ignored_symptoms`, prediction still runs on whatever's valid.
- If *every* symptom is unrecognized → `400` (nothing left to predict on).

### `GET /health`, `GET /`
Basic liveness/info checks (also report loaded model class/feature counts).

## 4. CORS
`allow_origins=["*"]` is enabled in `app/main.py` — fine for this
dev/thesis project since the Flutter app calls it from a device/emulator,
not a browser page on the same origin. Tighten this before any public
deployment.

## 5. Connecting the Flutter app

The Flutter project currently uses `DummyDataService` for everything
(see `lib/services/dummy_data_service.dart`). To connect it to this API:

1. **Add the `http` package** to `pubspec.yaml`:
   ```yaml
   dependencies:
     http: ^1.2.0
   ```

2. **Base URL** — Android emulator can't reach `localhost` directly:
   - Android emulator → `http://10.0.2.2:8000`
   - iOS simulator → `http://localhost:8000`
   - Physical device → your machine's LAN IP, e.g. `http://192.168.x.x:8000`

3. **Create `lib/services/api_service.dart`**:
   ```dart
   import 'dart:convert';
   import 'package:http/http.dart' as http;
   import 'package:smart_health_assistant/models/prediction_result.dart';

   class ApiService {
     static const String baseUrl = 'http://10.0.2.2:8000'; // adjust per platform

     static Future<List<Map<String, dynamic>>> getSymptoms() async {
       final res = await http.get(Uri.parse('$baseUrl/symptoms'));
       if (res.statusCode != 200) throw Exception('Failed to load symptoms');
       final data = jsonDecode(res.body);
       return List<Map<String, dynamic>>.from(data['symptoms']);
     }

     static Future<PredictionResult> predict(List<String> symptoms) async {
       final res = await http.post(
         Uri.parse('$baseUrl/predict'),
         headers: {'Content-Type': 'application/json'},
         body: jsonEncode({'symptoms': symptoms}),
       );
       if (res.statusCode != 200) {
         final err = jsonDecode(res.body);
         throw Exception(err['detail'] ?? 'Prediction failed');
       }
       return PredictionResult.fromJson(jsonDecode(res.body));
     }
   }
   ```
   Because `/predict`'s response uses the exact same keys as
   `PredictionResult.fromJson` already expects, this is a drop-in
   replacement for `DummyDataService.instance.analyzeSymptoms(...)` — no
   changes needed to `result_screen.dart` or `ResultCard`.

4. **Symptom list mismatch to be aware of**: `lib/utils/constants.dart`
   currently hardcodes a small, lowercase symptom list
   (`symptomCategoriesMap`) totally separate from the model's real 374
   symptoms in `symptom_dictionary.csv` (capitalized, e.g. "Runny Nose"
   not "Runny nose", no category grouping). `symptom_selection_screen.dart`
   builds its checklist from `DummyDataService.getSymptoms()`, which reads
   `constants.dart`. To use the real symptom set:
   - Replace `getSymptoms()` with a call to `ApiService.getSymptoms()`,
     mapping each `{id, name, severity_level}` into a `SymptomModel`
     (`category` can default to `"General"` or be dropped from the UI,
     since the real dictionary has no category column; `isEmergency` can
     be set from `severity_level == "Severe"`).
   - Send back the *exact* `name` strings the user selected in
     `POST /predict` — matching is exact string match, no fuzzy lookup.

## 6. Deployment notes
- The whole `data/` folder (~1.8 MB) ships with the backend; no external
  storage needed.
- For a single always-on server: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
  behind a process manager (e.g. `systemd`, `pm2`, or a Docker container
  running the same command) is enough for a thesis/demo deployment.
- If you deploy on a free host (Render, Railway, etc.), just make sure the
  `data/` folder is included in the build — it's not fetched from
  anywhere at runtime.
