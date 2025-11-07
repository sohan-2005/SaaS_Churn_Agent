from fastapi import FastAPI, UploadFile, File, HTTPException
from features import compute_features
from explain import predict_proba, explain_instance
from rag_utils import ingest_notes, retrieve_relevant
import pandas as pd, os, joblib, json

from google import genai
from dotenv import load_dotenv

app = FastAPI()

# Paths
STORAGE = "storage"
LATEST_CSV = os.path.join(STORAGE, "latest_uploaded.csv")
LATEST_FEATURES = os.path.join(STORAGE, "latest_features.csv")
MODEL_PATH = os.path.join(STORAGE, "churn_model.pkl")

load_dotenv()

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    content = await file.read()
    os.makedirs(STORAGE, exist_ok=True)
    with open(LATEST_CSV, "wb") as f:
        f.write(content)

    try:
        df = pd.read_csv(LATEST_CSV)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid CSV format")

    feats = compute_features(df)
    feats.to_csv(LATEST_FEATURES, index=False)
    return {"status": "ok", "rows": len(feats)}

@app.get("/predict")
def predict_all():
    if not os.path.exists(LATEST_FEATURES):
        raise HTTPException(404, "No features found. Upload CSV first.")
    feats = pd.read_csv(LATEST_FEATURES)
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(500, "Model not found. Run model_train.py first.")
    probs = predict_proba(feats)
    feats["churn_prob"] = probs
    return {"predictions": feats[["customer_id", "churn_prob"]].to_dict(orient="records")}

@app.post("/explain/{customer_id}")
def explain_customer(customer_id: str):
    load_dotenv()

    if not os.path.exists(LATEST_FEATURES):
        raise HTTPException(404, "No features found. Upload CSV first.")
    feats = pd.read_csv(LATEST_FEATURES)
    row = feats[feats["customer_id"] == customer_id]
    if row.empty:
        raise HTTPException(404, "Customer not found")

    p = float(predict_proba(row)[0])
    drivers = explain_instance(row.iloc[0])
    history = retrieve_relevant(customer_id)

    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        feature_summary = ", ".join([f"{f} ({round(v,2)})" for f, v in drivers])
        history_text = " ".join([n["text"] for n in history]) if history else "No prior notes."
        prompt = f"""
        You are an AI analyst helping a SaaS company interpret churn risk.
        Given this data:
        - Churn probability: {p:.2f}
        - Feature contributions: {feature_summary}
        - Customer history: {history_text}

        Write 2-3 concise sentences explaining why this customer's churn risk is {p:.2f}.
        Use clear, professional language.
        """

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        natural_text = response.text.strip()
    except Exception as e:
        natural_text = f"(LLM explanation unavailable: {str(e)})"

    return {
        "customer_id": customer_id,
        "churn_prob": p,
        "top_drivers": drivers,
        "history": history,
        "gemini_summary": natural_text,
    }


@app.post("/ingest_notes")
def ingest(notes: dict):
    note_list = notes.get("notes", [])
    ingest_notes(note_list)
    return {"status": "ok", "ingested": len(note_list)}