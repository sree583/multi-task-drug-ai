import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors

def compute_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def compute_basic_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "MolWt": np.nan,
            "LogP": np.nan,
            "HBD": np.nan,
            "HBA": np.nan,
            "TPSA": np.nan,
            "RotBonds": np.nan,
        }
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "TPSA": Descriptors.TPSA(mol),
        "RotBonds": Descriptors.NumRotatableBonds(mol),
    }

def build_feature_matrix(smiles_list, use_descriptors=True, radius=2, n_bits=2048):
    fps = []
    descs = []

    for s in smiles_list:
        fp = compute_morgan_fingerprint(s, radius=radius, n_bits=n_bits)
        fps.append(fp)
        if use_descriptors:
            d = compute_basic_descriptors(s)
            descs.append(d)

    X_fp = np.vstack(fps, dtype=np.float32)

    if use_descriptors:
        desc_df = pd.DataFrame(descs)
        desc_df = desc_df.fillna(desc_df.median())
        X_desc = desc_df.values.astype(np.float32)
        X = np.hstack([X_fp, X_desc])
    else:
        X = X_fp

    return X
