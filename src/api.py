from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
import joblib
from .features import build_feature_matrix
from .model import MultiTaskDrugNet

app = FastAPI(title="Multi‑Task Drug AI API")

MODEL_PATH = "output/models/multitask_model.pt"
CFG_PATH = "output/models/feature_config.joblib"

device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = joblib.load(CFG_PATH)

input_dim = 2048 + 6  # fingerprint + 6 descriptors
model = MultiTaskDrugNet(input_dim, hidden_dim=512).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

class SmilesInput(BaseModel):
    smiles: str

@app.post("/predict")
def predict(inp: SmilesInput):
    X = build_feature_matrix([inp.smiles], use_descriptors=cfg["use_descriptors"])
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

    with torch.no_grad():
        pIC50, hepatotox, ames, bbb, sol = model(X_tensor)

    pIC50_val = pIC50.item()
    hepatotox_prob = torch.softmax(hepatotox, dim=1)[0, 1].item()
    ames_prob = torch.softmax(ames, dim=1)[0, 1].item()
    bbb_prob = torch.softmax(bbb, dim=1)[0, 1].item()
    sol_class = int(sol.argmax(dim=1).item())

    return {
        "smiles": inp.smiles,
        "predicted_pIC50": round(pIC50_val, 3),
        "hepatotox_risk": round(hepatotox_prob, 3),
        "ames_risk": round(ames_prob, 3),
        "bbb_penetrant_prob": round(bbb_prob, 3),
        "solubility_class": sol_class,  # 0=low,1=medium,2=high
    }
