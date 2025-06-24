# Oxford-102 Flower Classification API

This repository contains a FastAPI microservice and an optional Streamlit app for classifying images from the Oxford-102 flower dataset using a fine-tuned ResNet50 model.

## Files

- `app.py` – FastAPI application exposing a `/predict` endpoint.
- `streamlit_app.py` – Streamlit application for interactive image uploads (optional).
- `requirements.txt` – Python dependencies.
- `.gitignore` – Files and directories to ignore in Git.
- `README.md` – This file.

## Prerequisites

- Python 3.10
- Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

- Install dependencies:

```bash
pip install -r requirements.txt
```

## Running Locally

### FastAPI

```bash
uvicorn app:app --reload
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to explore the API.

### Streamlit

```bash
streamlit run streamlit_app.py
```

Open the link provided by Streamlit to interact with the app.

## Deployment

### Render.com

1. Push this repository to GitHub.  
2. Create a new Web Service on Render, link your repo.  
3. Set Build Command: `pip install -r requirements.txt`  
4. Set Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

### Google Cloud Run

1. Add a `Dockerfile`.  
2. Build & push:  
   ```bash
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/flower-api
   ```  
3. Deploy:  
   ```bash
   gcloud run deploy flower-api      --image gcr.io/YOUR_PROJECT_ID/flower-api      --platform managed      --allow-unauthenticated
   ```

## API Usage

**POST** `/predict`  
- Form field: `file` (image)  
- Returns:
  ```json
  {
    "predicted_class_index": 81,
    "flower_name": "rose",
    "confidence": 0.87
  }
  ```

## License

MIT
