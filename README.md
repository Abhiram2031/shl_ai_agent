
# SHL Conversational Assessment Agent

## Features
- FastAPI backend
- Stateless conversational API
- Recommendation endpoint
- Health endpoint
- Dataset-ready architecture
- SHL assessment recommendation flow
- Easy VS Code setup

## Run Project

### Create virtual environment
```bash
python -m venv venv
```

### Activate
Windows:
```bash
venv\Scripts\activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run API
```bash
uvicorn app.main:app --reload
```

## Swagger Docs
http://127.0.0.1:8000/docs

## Deploy on Render

This repo includes `render.yaml` for a Render Web Service deployment.

### 1) Push code to GitHub

```bash
git add .
git commit -m "Add Render deployment config"
git push
```

### 2) Create service in Render

- In Render, click **New +** -> **Blueprint**.
- Connect your GitHub repo and select this repository.
- Render reads `render.yaml` and creates `shl-agent-api`.

### 3) Set environment variable

- In Render service settings, add:
  - `GROQ_API_KEY=<your_groq_api_key>`

### 4) Deploy and verify routes

After deploy, Render gives a public base URL:

`https://<your-service>.onrender.com`

Verify:

```bash
curl https://<your-service>.onrender.com/health
```

```bash
curl -X POST "https://<your-service>.onrender.com/chat" \
  -H "Content-Type: application/json" \
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"I need assessments for entry level sales roles\"}]}"
```

Expected:
- `/health` returns `{"status":"ok"}`
- `/chat` returns JSON with `reply`, `recommendations`, and `end_of_conversation`

## Dataset
Add your dataset here:

data/shl_catalog.json

Example:
[
  {
    "name": "Java 8 (New)",
    "url": "https://www.shl.com",
    "description": "Java assessment",
    "test_type": "K"
  }
]
