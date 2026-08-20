import os
import torch
import numpy as np
import pandas as pd
import joblib
from .data_utils import fetch_chembl_activity, clean_and_prepare_potency_data, add_synthetic_admet_labels
from .features import build_feature_matrix
from .model import train_multitask_model

def run_training(target_chembl_id, output_dir, hidden_dim=512, epochs=20, device="cpu"):
    os.makedirs(output_dir, exist_ok=True)
    data_dir = os.path.join(output_dir, "data")
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. Fetch potency data
    df_raw = fetch_chembl_activity(target_chembl_id)
    df_pot = clean_and_prepare_potency_data(df_raw)

    # 2. Add synthetic ADMET labels (replace later with real data)
    df = add_synthetic_admet_labels(df_pot)

    data_path = os.path.join(data_dir, "multitask_data.csv")
    df.to_csv(data_path, index=False)

    smiles_list = df["canonical_smiles"].tolist()
    y_potency = df["pIC50"].values.astype(np.float32)
    y_admet = {
        "hepatotox": df["hepatotox"].values.astype(np.int64),
        "ames": df["ames"].values.astype(np.int64),
        "bbb": df["bbb"].values.astype(np.int64),
        "solubility_class": df["solubility_class"].values.astype(np.int64),
    }

    X = build_feature_matrix(smiles_list, use_descriptors=True).astype(np.float32)

    # 3. Train model
    model, metrics, splits = train_multitask_model(
        X, y_potency, y_admet,
        hidden_dim=hidden_dim, epochs=epochs, device=device
    )

    # 4. Save model and metadata
    model_path = os.path.join(models_dir, "multitask_model.pt")
    torch.save(model.state_dict(), model_path)

    # Save feature config for inference
    feature_cfg = {
        "use_descriptors": True,
        "radius": 2,
        "n_bits": 2048,
        "smiles_column": "canonical_smiles",
        "target_chembl_id": target_chembl_id,
    }
    cfg_path = os.path.join(models_dir, "feature_config.joblib")
    joblib.dump(feature_cfg, cfg_path)

    # Save metrics
    metrics_path = os.path.join(models_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Target: {target_chembl_id}\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    return metrics
