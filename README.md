# SaaS Churn Predictor + Explainer Agent (Hackathon Starter)

Files:
- main.py          : FastAPI backend with endpoints /upload, /predict, /explain/{customer_id}
- app.py           : Streamlit frontend to upload CSV, view predictions and explanations
- features.py      : Feature extraction utilities
- explain.py       : Model inference + SHAP-based explanation (uses a lightweight sklearn model for demo)
- rag_utils.py     : Simple RAG-like retrieval helper (placeholder)
- model_train.py   : Script that trains a tiny demo model and saves to storage/churn_model.pkl
- storage/         : Contains churn_model.pkl, memory.json, and tmp files
- input.json/output.json : AgenThink contract files
- requirements.txt : Dependencies list (for hackathon, additional libs may be installed on platform)

Notes:
- This package includes a small synthetic dataset and a demo logistic regression model so you can run a working demo locally without heavy training.
- To run locally:
  1. Create and activate a virtualenv with Python 3.10+
  2. pip install -r requirements.txt
  3. Start backend: `uvicorn main:app --reload`
  4. Start frontend: `streamlit run app.py`
  5. Open Streamlit at http://localhost:8501

AgenThink:
- Upload the folder or zip on AgenThink; set `/explain/{customer_id}` or `/predict` as primary endpoint.
