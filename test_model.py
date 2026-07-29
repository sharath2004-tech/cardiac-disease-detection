"""
Quick model test on 3 real PTB-XL ECG patients (NORMAL, MI, HYP).
Run: py test_model.py
"""
import ast
import pickle

import numpy as np
import pandas as pd
import torch
import wfdb

from cardiom3net.config import Config
from cardiom3net.models.cardiom3net import CardioM3Net

cfg = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT = 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'

# ── Load model ────────────────────────────────────────────────────────
ckpt = torch.load(
    'cardiom3net_results/cardiom3net_central.pth',
    map_location=device, weights_only=False,
)
model = CardioM3Net(
    num_leads=12, clinical_dim=7,
    ecg_embed_dim=cfg.ecg_embed_dim,
    clinical_embed_dim=cfg.clinical_embed_dim,
    fusion_dim=cfg.fusion_dim,
    transformer_heads=cfg.transformer_heads,
    transformer_layers=cfg.transformer_layers,
    transformer_ff_dim=cfg.transformer_ff_dim,
    num_disease_classes=5, num_severity_classes=3,
    num_domains=3, da_lambda=cfg.da_lambda, dropout=cfg.dropout,
).to(device)
model.load_state_dict(ckpt['state_dict'])
model.eval()

imputer = pickle.load(open('cardiom3net_results/clinical_imputer.pkl', 'rb'))
scaler  = pickle.load(open('cardiom3net_results/clinical_scaler.pkl',  'rb'))

# ── Pick 3 PTB-XL patients with known labels ─────────────────────────
ptbxl = pd.read_csv(f'{ROOT}/ptbxl_database.csv', index_col='ecg_id')
ptbxl['scp_codes'] = ptbxl['scp_codes'].apply(ast.literal_eval)

test_ids = {
    'NORMAL': ptbxl[ptbxl['scp_codes'].apply(lambda x: 'NORM' in x)].index[0],
    'MI'    : ptbxl[ptbxl['scp_codes'].apply(lambda x: 'IMI' in x or 'AMI' in x)].index[0],
    'HYP'   : ptbxl[ptbxl['scp_codes'].apply(lambda x: 'LVH' in x)].index[0],
}

CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
SEVS    = ['Mild', 'Moderate', 'Severe']

print(f"\n{'Patient':<10} {'True Label':<12} {'Predicted':<12} {'Disease Prob':<14} Severity")
print('-' * 62)

for true_label, ecg_id in test_ids.items():
    row = ptbxl.loc[ecg_id]

    # Load & normalise ECG
    signal, _ = wfdb.rdsamp(f"{ROOT}/{row['filename_lr']}")
    signal = signal[:1000].T.astype(np.float32)
    signal = (signal - signal.mean(1, keepdims=True)) / (signal.std(1, keepdims=True) + 1e-6)

    # Clinical features: age, sex, height, weight, nurse, site, device_code
    clin_raw = np.array([[
        row.get('age', 60), row.get('sex', 1),
        row.get('height', 170), row.get('weight', 75),
        row.get('nurse', 1), row.get('site', 1), 1,
    ]], dtype=np.float32)

    clin = torch.tensor(scaler.transform(imputer.transform(clin_raw))).to(device)
    ecg  = torch.tensor(signal).unsqueeze(0).to(device)

    with torch.no_grad():
        preds, _, _ = model(ecg, clin)
        prob     = torch.softmax(preds['binary'], dim=1)[0, 1].item()
        disease  = preds['disease'].argmax(1).item()
        severity = preds['severity'].argmax(1).item()

    correct = '✓' if (true_label == 'NORMAL' and disease == 0) or \
                     (true_label != 'NORMAL' and disease != 0) else '✗'
    print(f"{true_label:<10} {true_label:<12} {CLASSES[disease]:<12} {prob:.1%}         "
          f"{SEVS[severity]}  {correct}")

print()
