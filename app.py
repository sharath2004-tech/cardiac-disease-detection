"""
CardioM3Net — Web Test Interface
Run:  py app.py
Open: http://localhost:5050
"""
import ast
import os
import pickle
import sys

# Ensure all relative paths resolve from the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import wfdb
from flask import Flask, jsonify, render_template_string, request

from cardiom3net.config import Config
from cardiom3net.models.cardiom3net import CardioM3Net

# ── Load model once at startup ────────────────────────────────────────
cfg    = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT   = 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'

print(f"Loading model on {device}...")
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

# Load PTB-XL index for record lookup
ptbxl = pd.read_csv(f'{ROOT}/ptbxl_database.csv', index_col='ecg_id')
ptbxl['scp_codes'] = ptbxl['scp_codes'].apply(ast.literal_eval)

if 'device' in ptbxl.columns:
    ptbxl['device_code'] = pd.Categorical(ptbxl['device']).codes.astype(float)
else:
    ptbxl['device_code'] = 0.0

CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
CLASS_NAMES = {
    'NORM': 'Normal',
    'MI':   'Myocardial Infarction',
    'STTC': 'ST/T-wave Change',
    'CD':   'Conduction Disorder',
    'HYP':  'Hypertrophy',
}
SEVS = ['Mild', 'Moderate', 'Severe']
print("Model ready. Open http://localhost:5050")

# ── HTML Template ─────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CardioM3Net — ECG Analysis</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f0f4ff; min-height: 100vh; }
  .header { background: linear-gradient(135deg, #1a2d5a, #2563eb);
            color: white; padding: 24px 32px; }
  .header h1 { font-size: 24px; font-weight: 700; }
  .header p  { font-size: 14px; opacity: 0.8; margin-top: 4px; }
  .container { max-width: 860px; margin: 32px auto; padding: 0 16px; }
  .card { background: white; border-radius: 12px; padding: 28px;
          box-shadow: 0 2px 12px rgba(0,0,0,0.08); margin-bottom: 24px; }
  .card h2 { font-size: 16px; font-weight: 600; color: #1a2d5a;
             margin-bottom: 20px; padding-bottom: 10px;
             border-bottom: 2px solid #dbeafe; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .field label { display: block; font-size: 12px; font-weight: 600;
                 color: #6b7280; margin-bottom: 6px; text-transform: uppercase;
                 letter-spacing: 0.05em; }
  .field input, .field select {
    width: 100%; padding: 10px 14px; border: 1.5px solid #e5e7eb;
    border-radius: 8px; font-size: 14px; color: #111827;
    transition: border 0.2s; outline: none; }
  .field input:focus, .field select:focus { border-color: #2563eb; }
  .hint { font-size: 11px; color: #9ca3af; margin-top: 4px; }
  .ecg-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .btn { width: 100%; padding: 14px; background: linear-gradient(135deg,#2563eb,#1d4ed8);
         color: white; border: none; border-radius: 10px; font-size: 16px;
         font-weight: 600; cursor: pointer; transition: opacity 0.2s; }
  .btn:hover { opacity: 0.9; }
  .btn:disabled { opacity: 0.6; cursor: not-allowed; }
  .result { display: none; }
  .result.show { display: block; }
  .risk-bar-wrap { background: #f3f4f6; border-radius: 999px; height: 22px;
                   overflow: hidden; margin: 10px 0; }
  .risk-bar { height: 100%; border-radius: 999px; display: flex;
              align-items: center; justify-content: flex-end;
              padding-right: 10px; color: white; font-size: 12px;
              font-weight: 700; transition: width 0.8s ease; }
  .metric-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; }
  .metric { background: #f8faff; border: 1.5px solid #dbeafe; border-radius: 10px;
            padding: 16px; text-align: center; }
  .metric .val { font-size: 22px; font-weight: 700; color: #1a2d5a; }
  .metric .lbl { font-size: 11px; color: #6b7280; margin-top: 4px;
                 text-transform: uppercase; letter-spacing: 0.05em; }
  .class-bars { margin-top: 16px; }
  .class-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
  .class-name { width: 190px; font-size: 12px; color: #374151; flex-shrink: 0; }
  .class-track { flex: 1; background: #f3f4f6; border-radius: 999px; height: 14px; overflow: hidden; }
  .class-fill { height: 100%; border-radius: 999px; transition: width 0.8s ease; }
  .class-pct { width: 42px; text-align: right; font-size: 12px;
               color: #6b7280; flex-shrink: 0; }
  .badge { display: inline-block; padding: 4px 12px; border-radius: 999px;
           font-size: 12px; font-weight: 600; }
  .badge-green  { background: #dcfce7; color: #15803d; }
  .badge-yellow { background: #fef9c3; color: #a16207; }
  .badge-red    { background: #fee2e2; color: #b91c1c; }
  .badge-blue   { background: #dbeafe; color: #1d4ed8; }
  .error-box { background: #fee2e2; border: 1.5px solid #fca5a5;
               border-radius: 8px; padding: 12px 16px; color: #b91c1c;
               font-size: 13px; margin-top: 12px; display: none; }
  .spinner { display: none; text-align: center; padding: 12px;
             color: #2563eb; font-size: 13px; }
  .true-label { background: #f0fdf4; border: 1.5px solid #86efac;
                border-radius: 8px; padding: 10px 14px; font-size: 13px;
                color: #15803d; margin-top: 12px; }
  @media(max-width:600px) {
    .grid,.ecg-row,.metric-grid { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>

<div class="header">
  <h1>&#9829; CardioM3Net — ECG Analysis</h1>
  <p>12-Lead ECG · Multi-Task Cardiac Disease Detection · PTB-XL Dataset</p>
</div>

<div class="container">

  <!-- Input Card -->
  <div class="card">
    <h2>Patient & ECG Input</h2>

    <div class="ecg-row" style="margin-bottom:20px;">
      <div class="field">
        <label>ECG Record ID (PTB-XL)</label>
        <input type="number" id="ecg_id" value="1" min="1" max="21837" placeholder="e.g. 1, 500, 2000">
        <div class="hint">Patient ID from PTB-XL (1 – 21837). The ECG file will be loaded from disk.</div>
      </div>
      <div class="field" style="display:flex;flex-direction:column;justify-content:flex-end;">
        <button class="btn" id="autoFillBtn" style="background:linear-gradient(135deg,#059669,#047857);font-size:13px;padding:10px;"
                onclick="autoFill()">&#x2728; Auto-fill from PTB-XL record</button>
      </div>
    </div>

    <div class="grid">
      <div class="field">
        <label>Age (years)</label>
        <input type="number" id="age" value="55" min="1" max="120">
      </div>
      <div class="field">
        <label>Sex</label>
        <select id="sex">
          <option value="1">Male</option>
          <option value="0">Female</option>
        </select>
      </div>
      <div class="field">
        <label>Height (cm)</label>
        <input type="number" id="height" value="170" min="100" max="220">
      </div>
      <div class="field">
        <label>Weight (kg)</label>
        <input type="number" id="weight" value="75" min="30" max="200">
      </div>
    </div>

    <div style="margin-top:20px;">
      <button class="btn" onclick="predict()">&#x1F50D; Analyse ECG</button>
    </div>
    <div class="spinner" id="spinner">Analysing ECG signal...</div>
    <div class="error-box" id="errorBox"></div>
  </div>

  <!-- Result Card -->
  <div class="card result" id="resultCard">
    <h2>Analysis Results <span id="ecgIdBadge" class="badge badge-blue"></span></h2>

    <div id="trueLabel" class="true-label" style="display:none;"></div>

    <!-- Risk bar -->
    <div style="margin:20px 0 10px;">
      <div style="font-size:13px;font-weight:600;color:#374151;margin-bottom:6px;">
        Disease Probability
        <span id="probBadge" class="badge" style="margin-left:8px;"></span>
      </div>
      <div class="risk-bar-wrap">
        <div class="risk-bar" id="riskBar"></div>
      </div>
    </div>

    <!-- Key metrics -->
    <div class="metric-grid">
      <div class="metric">
        <div class="val" id="probVal">—</div>
        <div class="lbl">Disease Probability</div>
      </div>
      <div class="metric">
        <div class="val" id="classVal">—</div>
        <div class="lbl">Disease Class</div>
      </div>
      <div class="metric">
        <div class="val" id="sevVal">—</div>
        <div class="lbl">Severity</div>
      </div>
    </div>

    <!-- Class probabilities -->
    <div class="class-bars" style="margin-top:24px;">
      <div style="font-size:13px;font-weight:600;color:#374151;margin-bottom:12px;">
        Class Probabilities
      </div>
      <div id="classBars"></div>
    </div>
  </div>

</div>

<script>
const CLASSES = ['NORM','MI','STTC','CD','HYP'];
const CLASS_NAMES = {
  NORM:'Normal', MI:'Myocardial Infarction',
  STTC:'ST/T-wave Change', CD:'Conduction Disorder', HYP:'Hypertrophy'
};
const COLORS = ['#2563eb','#dc2626','#f59e0b','#7c3aed','#059669'];
const SEV_COLORS = {Mild:'#059669', Moderate:'#f59e0b', Severe:'#dc2626'};

async function autoFill() {
  const ecg_id = document.getElementById('ecg_id').value;
  const btn = document.getElementById('autoFillBtn');
  btn.disabled = true; btn.textContent = 'Loading...';
  try {
    const r = await fetch(`/patient_info?ecg_id=${ecg_id}`);
    const d = await r.json();
    if (d.error) { alert(d.error); return; }
    if (d.age)    document.getElementById('age').value    = d.age;
    if (d.sex !== undefined) document.getElementById('sex').value = d.sex;
    if (d.height) document.getElementById('height').value = d.height;
    if (d.weight) document.getElementById('weight').value = d.weight;
  } catch(e) { alert('Failed to load patient info'); }
  finally { btn.disabled = false; btn.textContent = '✨ Auto-fill from PTB-XL record'; }
}

async function predict() {
  const ecg_id  = document.getElementById('ecg_id').value;
  const age     = document.getElementById('age').value;
  const sex     = document.getElementById('sex').value;
  const height  = document.getElementById('height').value;
  const weight  = document.getElementById('weight').value;

  document.getElementById('errorBox').style.display  = 'none';
  document.getElementById('spinner').style.display   = 'block';
  document.getElementById('resultCard').classList.remove('show');

  try {
    const r = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ecg_id:+ecg_id, age:+age, sex:+sex,
                            height:+height, weight:+weight})
    });
    const d = await r.json();
    if (d.error) throw new Error(d.error);
    showResult(d, ecg_id);
  } catch(e) {
    const eb = document.getElementById('errorBox');
    eb.textContent = 'Error: ' + e.message;
    eb.style.display = 'block';
  } finally {
    document.getElementById('spinner').style.display = 'none';
  }
}

function showResult(d, ecg_id) {
  const prob = d.disease_probability;
  const pct  = Math.round(prob * 100);

  document.getElementById('ecgIdBadge').textContent = `ECG #${ecg_id}`;
  document.getElementById('probVal').textContent  = pct + '%';
  document.getElementById('classVal').textContent = d.predicted_class;
  document.getElementById('sevVal').textContent   = d.severity;

  // Risk bar colour
  const col = prob < 0.3 ? '#16a34a' : prob < 0.6 ? '#f59e0b' : '#dc2626';
  const bar = document.getElementById('riskBar');
  bar.style.background = col;
  bar.style.width = pct + '%';
  bar.textContent = pct + '%';

  // Badge
  const badge = document.getElementById('probBadge');
  if (prob < 0.3) { badge.className='badge badge-green'; badge.textContent='Low Risk'; }
  else if (prob < 0.6) { badge.className='badge badge-yellow'; badge.textContent='Moderate Risk'; }
  else { badge.className='badge badge-red'; badge.textContent='High Risk'; }

  // Severity in metric
  const sevEl = document.getElementById('sevVal');
  sevEl.style.color = SEV_COLORS[d.severity] || '#1a2d5a';

  // True label if available
  const tlEl = document.getElementById('trueLabel');
  if (d.true_label) {
    tlEl.innerHTML = `&#x2714; <b>Ground Truth (PTB-XL):</b> ${d.true_label}`;
    tlEl.style.display = 'block';
  } else {
    tlEl.style.display = 'none';
  }

  // Class probability bars
  const barsEl = document.getElementById('classBars');
  barsEl.innerHTML = '';
  CLASSES.forEach((cls, i) => {
    const p = d.class_probs[cls] || 0;
    const isPred = cls === d.predicted_class_code;
    barsEl.innerHTML += `
      <div class="class-row">
        <div class="class-name">
          ${isPred ? '&#x25B6; ' : ''}<b>${cls}</b> — ${CLASS_NAMES[cls]}
        </div>
        <div class="class-track">
          <div class="class-fill" style="width:${(p*100).toFixed(1)}%;background:${COLORS[i]};
               ${isPred?'opacity:1':'opacity:0.5'}"></div>
        </div>
        <div class="class-pct">${(p*100).toFixed(1)}%</div>
      </div>`;
  });

  document.getElementById('resultCard').classList.add('show');
  document.getElementById('resultCard').scrollIntoView({behavior:'smooth'});
}
</script>
</body>
</html>
"""

app = Flask(__name__)


def load_ecg(ecg_id: int):
    """Load and normalise a PTB-XL ECG by ecg_id."""
    if ecg_id not in ptbxl.index:
        return None, f"ECG ID {ecg_id} not found in PTB-XL"
    row = ptbxl.loc[ecg_id]
    try:
        signal, _ = wfdb.rdsamp(f"{ROOT}/{row['filename_lr']}")
        signal = signal[:1000].T.astype(np.float32)
        signal = (signal - signal.mean(1, keepdims=True)) / (signal.std(1, keepdims=True) + 1e-6)
        return signal, None
    except Exception as e:
        return None, str(e)


def get_true_label(ecg_id: int) -> str | None:
    """Return human-readable ground-truth label from SCP codes."""
    if ecg_id not in ptbxl.index:
        return None
    scp = ptbxl.loc[ecg_id, 'scp_codes']
    codes = list(scp.keys())
    for c in ['NORM', 'MI', 'STTC', 'CD', 'HYP']:
        if any(c in k for k in codes):
            return f"{c} — {CLASS_NAMES[c]}"
    return ', '.join(codes[:3]) if codes else None


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/patient_info')
def patient_info():
    try:
        ecg_id = int(request.args.get('ecg_id', 1))
        if ecg_id not in ptbxl.index:
            return jsonify({'error': f'ECG ID {ecg_id} not found'})
        row = ptbxl.loc[ecg_id]
        return jsonify({
            'age':    float(row.get('age', '')) if not pd.isna(row.get('age', float('nan'))) else None,
            'sex':    int(row.get('sex', 1)),
            'height': float(row.get('height', '')) if not pd.isna(row.get('height', float('nan'))) else None,
            'weight': float(row.get('weight', '')) if not pd.isna(row.get('weight', float('nan'))) else None,
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data   = request.get_json()
        ecg_id = int(data.get('ecg_id', 1))
        age    = float(data.get('age', 55))
        sex    = float(data.get('sex', 1))
        height = float(data.get('height', 170))
        weight = float(data.get('weight', 75))

        # Load ECG
        signal, err = load_ecg(ecg_id)
        if err:
            return jsonify({'error': err})

        # Prepare clinical features
        clin_raw = np.array([[age, sex, height, weight, 1.0, 1.0, 1.0]], dtype=np.float32)
        clin = torch.tensor(
            scaler.transform(imputer.transform(clin_raw)),
            dtype=torch.float32,
        ).to(device)
        ecg_t = torch.tensor(signal).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            preds, _, _ = model(ecg_t, clin)
            bin_probs  = torch.softmax(preds['binary'],  dim=1)[0].cpu().numpy()
            dis_probs  = torch.softmax(preds['disease'], dim=1)[0].cpu().numpy()
            sev_probs  = torch.softmax(preds['severity'],dim=1)[0].cpu().numpy()

        disease_idx  = int(dis_probs.argmax())
        severity_idx = int(sev_probs.argmax())

        return jsonify({
            'disease_probability': float(bin_probs[1]),
            'predicted_class':     CLASS_NAMES[CLASSES[disease_idx]],
            'predicted_class_code': CLASSES[disease_idx],
            'severity':            SEVS[severity_idx],
            'class_probs':         {c: float(dis_probs[i]) for i, c in enumerate(CLASSES)},
            'true_label':          get_true_label(ecg_id),
        })

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=False)
