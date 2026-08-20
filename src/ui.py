import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.title("Multi‑Task Drug AI – Potency + ADMET Scorecard")

smiles = st.text_area("Enter one or more SMILES (one per line)", 
                      "CC(=O)Oc1ccccc1C(=O)O\nCC1=C(C(=O)O)N=C(O)C(=O)O1")

if st.button("Predict"):
    lines = [s.strip() for s in smiles.strip().split("\n") if s.strip()]
    results = []
    for s in lines:
        try:
            resp = requests.post(API_URL, json={"smiles": s})
            resp.raise_for_status()
            results.append(resp.json())
        except Exception as e:
            results.append({"smiles": s, "error": str(e)})

    st.subheader("Results")
    for r in results:
        st.write(f"**SMILES:** `{r['smiles']}`")
        if "error" in r:
            st.error(f"Error: {r['error']}")
            continue

        st.write(f"- Predicted pIC50: **{r['predicted_pIC50']}**")
        st.write(f"- Hepatotoxicity risk: **{r['hepatotox_risk']}**")
        st.write(f"- AMES mutagenicity risk: **{r['ames_risk']}**")
        st.write(f"- BBB penetrant probability: **{r['bbb_penetrant_prob']}**")
        st.write(f"- Solubility class: **{r['solubility_class']}** (0=low,1=medium,2=high)")
        st.markdown("---")
