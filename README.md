# multi-task-drug-ai
“Multi‑task deep learning for drug discovery: joint prediction of potency and ADMET/toxicity from SMILES. Includes training pipeline, FastAPI service, and Streamlit UI.”
# Multi‑Task Drug AI

Multi‑task deep learning model for small‑molecule drug discovery: jointly predicts **potency (pIC50)** and multiple **ADMET/toxicity** endpoints from SMILES.

## Features

- Data download from ChEMBL for a chosen target.
- Multi‑task neural network (PyTorch) for:
  - Regression: pIC50
  - Classification: toxicity / ADMET flags
- Proper train/val/test splits and metrics:
  - Potency: RMSE, R²
  - ADMET: ROC‑AUC per task
- FastAPI service for predictions.
- Streamlit UI to upload SMILES and get a “drug‑likeness scorecard”.

## Setup

```bash
pip install -r requirements.txt
```

## Train the model

Example for EGFR (ChEMBL1077):

```bash
python run_train.py --target CHEMBL1077 --output-dir output
```

This will:

- Save cleaned data to `output/data/`.
- Save the trained model to `output/models/multitask_model.pt`.
- Print performance metrics to the console and to `output/models/metrics.txt`.

## Run the prediction API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)Oc1ccccc1C(=O)O"}'
```

## Run the Streamlit UI

```bash
streamlit run src/ui.py
```

Open the URL shown in the terminal (usually http://localhost:8501).

## Results to show in your portfolio

- Command‑prompt output: RMSE, R², ROC‑AUC values.
- Screenshots of:
  - API JSON response.
  - Streamlit UI with a few example compounds and their risk scorecard.
- Short write‑up: problem, data, model, metrics, limitations, business impact.

## License

MIT
