# Corticode — Brain Tumor MRI Classifier (Web App)

A frontend + backend around your Hybrid ConvNeXtTiny–DenseNet121–CBAM model.
You upload an MRI slice in the browser, the browser sends it to a small Flask
API, the API runs your trained model, and the browser shows the predicted
class with confidence bars.

```
brain-tumor-cad/
├── backend/
│   ├── app.py              # Flask API — loads the model, runs predictions
│   ├── model_builder.py    # Architecture definition (fallback if you only saved weights)
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
└── README.md
```

## 0. What you need before you start

- Python 3.10–3.11 (TensorFlow 2.16+ supports these; 3.12 support is newer/spottier)
- Your **trained model file**, `Hybrid_ConvNeXt_DenseNet121_CBAM.keras`, exported
  from the notebook. This is *not* included here — you need to download it from
  wherever you trained (Kaggle, Colab, etc.) after training finishes. In the
  notebook it's saved automatically by the `ModelCheckpoint` callback.

## 1. Get your trained model file out of the notebook

If you trained on Kaggle:
1. After training finishes, look in the notebook's **Output** files panel (right
   sidebar) for `Hybrid_ConvNeXt_DenseNet121_CBAM.keras`.
2. Click it → **Download**.
3. Move the downloaded file into `backend/` in this project, right next to `app.py`.

If you trained on Colab: it'll be in the Colab file browser (left sidebar) or
wherever your `ModelCheckpoint(filepath=...)` pointed — download it the same way.

**Important — Keras 3 + Lambda layers:** your CBAM spatial-attention block uses
`Lambda(lambda x: tf.reduce_mean(...))`. Keras 3 blocks deserializing arbitrary
Lambda layers by default for security reasons. `app.py` already handles this
(`safe_mode=False` when loading), so you don't need to change anything — just
know why that flag is there if you ever load the model yourself in a notebook.

## 2. Set up the backend

```bash
cd backend
python -m venv venv

# Mac/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

pip install -r requirements.txt
```

Make sure `Hybrid_ConvNeXt_DenseNet121_CBAM.keras` is sitting in this `backend/`
folder (same level as `app.py`). Then run:

```bash
python app.py
```

You should see Flask start on `http://localhost:5000`. Sanity-check it:

```bash
curl http://localhost:5000/api/health
# {"status": "ok", "model_loaded": true, ...}
```

If `model_loaded` is `false`, the response includes a `load_error` telling you
why (usually: wrong filename, or the file isn't actually in `backend/`).

## 3. Run the frontend

The frontend is plain HTML/CSS/JS — no build step. Two easy ways to serve it:

**Option A — just open it:**
Double-click `frontend/index.html`. This works for most browsers, but some
browsers restrict `fetch()` from `file://` pages, so if uploads silently fail,
use Option B instead.

**Option B — tiny local server (recommended):**
```bash
cd frontend
python -m http.server 5500
```
Then open `http://localhost:5500` in your browser.

## 4. How the frontend and backend actually connect

- `frontend/script.js` has one line that matters: `const API_BASE = "http://localhost:5000";`
  Every request (`/api/health`, `/api/predict`) goes to that address.
- When you click **Analyze scan**, the browser sends the image as
  `multipart/form-data` to `POST /api/predict`.
- `backend/app.py` reads the file, resizes it to 224×224, applies the same
  `ConvNeXt preprocess_input` normalization used in training, runs
  `model.predict(...)`, and returns JSON:
  ```json
  {
    "predicted_class": "glioma",
    "predicted_label": "Glioma",
    "confidence": 0.94,
    "results": [ {"class": "glioma", "confidence": 0.94, ...}, ... ]
  }
  ```
- `flask-cors` is enabled in `app.py` so the browser (running on port 5500) is
  allowed to call the API (running on port 5000) — different ports count as
  different origins, and without CORS enabled the browser blocks the request.

If you deploy the two pieces to different domains later (see below), just
update `API_BASE` in `script.js` to point at wherever the backend actually lives.

## 5. Using the other models from your paper

Your paper's comparison table (`Table I`) references other single-backbone
models (ConvNeXt alone, DenseNet+CBAM alone, etc.) that you trained during
experimentation. If you want to serve one of those instead:

1. Export that model the same way (`model.save("your_model.keras")` in the notebook).
2. Drop it in `backend/`.
3. Change `MODEL_PATH` in `app.py` (or set the `MODEL_PATH` environment
   variable instead of editing the file):
   ```bash
   MODEL_PATH=densenet_cbam.keras python app.py
   ```
4. If a model's output has different classes or a different class order than
   `["glioma", "healthy", "meningioma", "pituitary"]`, update `CLASS_NAMES` and
   `CLASS_INFO` at the top of `app.py` to match — this list must be in the
   exact order Keras assigned during training (check `train_data.class_indices`
   in your notebook, which is alphabetical by default: glioma, healthy,
   meningioma, pituitary).

You can only serve **one model at a time** with this simple setup. If you want
a dropdown to switch models live in the UI, that's a reasonable next feature —
say the word and I'll add it.

## 6. Deploying it for real (optional)

For a class demo, running both locally (steps 2–3) is enough. If you want a
shareable link:

- **Backend:** Render, Railway, or Fly.io all support "deploy a Flask app from
  a repo" for free/cheap tiers. Watch your model file size — ConvNeXtTiny +
  DenseNet121 combined will likely be 150–250MB, which exceeds some free-tier
  git/storage limits. You may need Git LFS or to load the model from cloud
  storage (S3/GCS) at startup instead of committing it to the repo.
- **Frontend:** Netlify or Vercel — drag-and-drop the `frontend/` folder, no
  build step needed. Update `API_BASE` in `script.js` to your deployed
  backend's URL before deploying.
- **CORS:** if frontend and backend end up on different domains in production,
  double check `CORS(app)` in `app.py` — it currently allows all origins,
  which is fine for a demo but you'd normally restrict it
  (`CORS(app, origins=["https://your-frontend-domain.com"])`) for anything
  more permanent.

## 7. Troubleshooting

| Symptom | Likely cause |
|---|---|
| "Backend not reachable" in the UI | `app.py` isn't running, or it's on a different port than `API_BASE` |
| "Model not loaded" | `.keras` file isn't in `backend/`, or filename doesn't match `MODEL_PATH` |
| `load_model` throws a Lambda / deserialization error | Make sure you're on a recent `tensorflow` (2.16+); older versions handle this differently — try `tf.keras.models.load_model(path)` without `safe_mode` if you're on an older version |
| Prediction looks random / always the same class | Check `CLASS_NAMES` order matches your notebook's `train_data.class_indices` |
| CORS error in browser console | Confirm `flask-cors` installed and `CORS(app)` is present in `app.py` |

---

**Reminder:** this is a research/educational tool, not a certified diagnostic
device. Real clinical AI products (radiology triage tools etc.) go through
regulatory clearance (e.g. FDA/CE marking) that this project has not — the UI
says as much, and that disclaimer should stay visible in any demo or write-up.
