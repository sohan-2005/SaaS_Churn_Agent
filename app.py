import streamlit as st
import requests
import pandas as pd

BACKEND = "http://127.0.0.1:8000"

st.set_page_config(page_title="SaaS Churn Predictor + Explainer", layout="centered")
st.title("💼 SaaS Churn Predictor + Explainer (Demo)")

uploaded = st.file_uploader("📂 Upload user activity CSV", type=["csv"])

if uploaded:
    try:
        resp = requests.post(
            BACKEND + "/upload",
            files={"file": ("uploaded.csv", uploaded.getvalue(), "text/csv")}
        )
    except Exception as e:
        st.error(f"Upload error: {e}")
        resp = None

    if resp is not None and resp.ok:
        st.success("✅ Upload successful")

        pred = requests.get(BACKEND + "/predict").json()
        df = pd.DataFrame(pred["predictions"])

        st.write("### 📊 Predicted Churn Probabilities")
        st.dataframe(df)

        if not df.empty:
            sel = st.selectbox("Select a customer to explain", df["customer_id"].tolist())

            if st.button("🔍 Explain"):
                exp = requests.post(BACKEND + f"/explain/{sel}")

                if exp.ok:
                    data = exp.json()

                    summary = data.get("gemini_summary", "")
                    if summary:
                        st.markdown("### 🤖 AI-Generated Explanation")
                        st.info(summary)

                    st.subheader(f"Customer {data['customer_id']} — Churn Probability: {data['churn_prob']:.2f}")

                    drivers = data.get("top_drivers", [])
                    contrib_sum = sum(abs(v) for _, v in drivers) or 1e-6
                    normalized = [(f, round(v / contrib_sum * 100, 2)) for f, v in drivers]
                    df_exp = pd.DataFrame(normalized, columns=["Feature", "Influence (%)"])

                    def color_contrib(val):
                        color = "red" if val > 0 else "green"
                        return f"color: {color}; font-weight: bold;"

                    st.markdown("### 🧩 Feature Influence Table")
                    st.dataframe(df_exp.style.applymap(color_contrib, subset=["Influence (%)"]))

                    st.markdown("### 📈 Visual Explanation")
                    st.bar_chart(df_exp.set_index("Feature"))

                    st.markdown("### 🗒️ Relevant History Notes")
                    history = data.get("history", [])
                    if not history:
                        st.info("No notes found for this customer.")
                    else:
                        for n in history:
                            st.markdown(f"- {n.get('text','')}")

                    st.markdown("### ✏️ Add a New History Note")
                    new_note = st.text_area("Write a short note about this customer")

                    if st.button("💾 Save Note"):
                        payload = {
                            "notes": [
                                {"id": f"auto_{sel}", "customer_id": sel, "text": new_note}
                            ]
                        }
                        res = requests.post(BACKEND + "/ingest_notes", json=payload)
                        if res.ok:
                            st.success("Note added successfully!")
                        else:
                            st.error(res.text)

                else:
                    st.error(f"❌ Backend error: {exp.text}")

    else:
        st.error("❌ Upload failed or no response yet. Check backend connection.")
else:
    st.info("👆 Upload a sample CSV file to start the churn prediction process.")