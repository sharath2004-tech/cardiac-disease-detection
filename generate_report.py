"""
Generate a project report PDF for CardioM3Net.
Run: py generate_report.py
Output: CardioM3Net_Project_Report.pdf
"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, white, black
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Image, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# ── Colours ──────────────────────────────────────────────────────────
NAVY    = HexColor('#1a2d5a')
BLUE    = HexColor('#2563eb')
LBLUE   = HexColor('#dbeafe')
GREEN   = HexColor('#16a34a')
LGREEN  = HexColor('#dcfce7')
RED     = HexColor('#dc2626')
GRAY    = HexColor('#6b7280')
LGRAY   = HexColor('#f3f4f6')
WHITE   = white

W, H = A4

def build_pdf():
    import time
    out = f"CardioM3Net_Report_{int(time.time())}.pdf"
    doc = SimpleDocTemplate(
        out, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.5*cm, bottomMargin=2.5*cm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle('title', fontSize=26, textColor=white,
                                 fontName='Helvetica-Bold', alignment=TA_CENTER,
                                 spaceAfter=6, leading=32)
    subtitle_style = ParagraphStyle('subtitle', fontSize=13, textColor=LBLUE,
                                    fontName='Helvetica', alignment=TA_CENTER,
                                    spaceAfter=4)
    h1 = ParagraphStyle('h1', fontSize=16, textColor=NAVY,
                         fontName='Helvetica-Bold', spaceBefore=18, spaceAfter=8,
                         borderPad=4)
    h2 = ParagraphStyle('h2', fontSize=12, textColor=BLUE,
                         fontName='Helvetica-Bold', spaceBefore=12, spaceAfter=6)
    body = ParagraphStyle('body', fontSize=10, textColor=black,
                           fontName='Helvetica', leading=15, spaceAfter=6,
                           alignment=TA_JUSTIFY)
    bullet = ParagraphStyle('bullet', fontSize=10, textColor=black,
                             fontName='Helvetica', leading=14, leftIndent=16,
                             spaceAfter=3)
    caption = ParagraphStyle('caption', fontSize=8, textColor=GRAY,
                              fontName='Helvetica-Oblique', alignment=TA_CENTER,
                              spaceAfter=6)
    metric_label = ParagraphStyle('ml', fontSize=10, textColor=GRAY,
                                   fontName='Helvetica', alignment=TA_CENTER)
    metric_val = ParagraphStyle('mv', fontSize=20, textColor=NAVY,
                                 fontName='Helvetica-Bold', alignment=TA_CENTER)

    story = []

    # ── COVER PAGE ────────────────────────────────────────────────────
    cover_data = [[
        Table(
            [[Paragraph("CardioM3Net", title_style)],
             [Paragraph("Multimodal Multi-Task Cardiac Disease Detection", subtitle_style)],
             [Paragraph("Using ECG Signals &amp; Clinical Features", subtitle_style)],
             [Spacer(1, 0.3*cm)],
             [Paragraph("Project Report  ·  2026", ParagraphStyle('date', fontSize=11,
               textColor=HexColor('#93c5fd'), fontName='Helvetica', alignment=TA_CENTER))],
            ],
            colWidths=[15*cm],
            style=TableStyle([('BACKGROUND', (0,0), (-1,-1), NAVY),
                               ('ROWBACKGROUNDS', (0,0), (-1,-1), [NAVY]),
                               ('TOPPADDING', (0,0), (-1,-1), 12),
                               ('BOTTOMPADDING', (0,0), (-1,-1), 12),
                               ('LEFTPADDING', (0,0), (-1,-1), 20),
                               ('RIGHTPADDING', (0,0), (-1,-1), 20),
                              ])
        )
    ]]
    cover_table = Table(cover_data, colWidths=[15*cm])
    cover_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), NAVY),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 30),
        ('BOTTOMPADDING', (0,0), (-1,-1), 30),
        ('ROUNDEDCORNERS', [8,8,8,8]),
    ]))
    story.append(cover_table)
    story.append(Spacer(1, 0.8*cm))

    # Key metrics banner
    metrics = [
        ("93.5%", "Binary AUC"),
        ("86.7%", "Accuracy"),
        ("87.4%", "F1 Score"),
        ("65.5%", "5-Class Accuracy"),
        ("83.7%", "Severity Accuracy"),
    ]
    mdata = [[Paragraph(v, metric_val) for v, _ in metrics],
             [Paragraph(l, metric_label) for _, l in metrics]]
    mt = Table(mdata, colWidths=[2.9*cm]*5)
    mt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), LGRAY),
        ('GRID', (0,0), (-1,-1), 0.5, HexColor('#e5e7eb')),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('ROUNDEDCORNERS', [6,6,6,6]),
    ]))
    story.append(mt)
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=LBLUE))
    story.append(Spacer(1, 0.3*cm))

    # ── SECTION 1: OVERVIEW ───────────────────────────────────────────
    story.append(Paragraph("1. Project Overview", h1))
    story.append(Paragraph(
        "CardioM3Net (Cardiac Multimodal Multi-task Meta-learning Network) is an "
        "end-to-end deep learning pipeline for automated cardiac disease detection. "
        "The system fuses 12-lead ECG signals with patient clinical features "
        "(age, sex, height, weight, recording site) to simultaneously predict "
        "three clinical outcomes: binary disease presence, specific disease class, "
        "and severity.", body))
    story.append(Paragraph(
        "The project targets the real-world challenge of scarce labelled medical data "
        "and cross-site variability through a combination of self-supervised pretraining, "
        "meta-learning, domain adaptation, and federated learning.", body))

    story.append(Paragraph("Key Objectives", h2))
    objectives = [
        "Detect the presence of cardiac disease from 12-lead ECG (binary classification)",
        "Classify the specific disease type: NORM, MI, STTC, CD, or HYP (5-class)",
        "Estimate disease severity: Mild, Moderate, or Severe (3-class)",
        "Generalise across recording sites and devices using domain adaptation",
        "Enable fast adaptation to new hospitals with very few labelled samples (MAML)",
        "Support privacy-preserving training across distributed sites (Federated Learning)",
    ]
    for obj in objectives:
        story.append(Paragraph(f"• {obj}", bullet))

    # ── SECTION 2: DATASET ────────────────────────────────────────────
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("2. Dataset", h1))
    story.append(Paragraph(
        "The model was trained on the <b>PTB-XL dataset</b> — the largest publicly "
        "available clinical 12-lead ECG dataset, released by Physionet.", body))

    ds_data = [
        ['Property', 'Value'],
        ['Source', 'PTB-XL (Physionet)'],
        ['Total Records', '21,837 ECG recordings'],
        ['Used in Training', '20,432 (balanced sampling)'],
        ['Train / Test Split', '16,345 train  /  4,087 test (80/20)'],
        ['Sampling Rate', '100 Hz  (1,000 timesteps per record)'],
        ['ECG Leads', '12 leads per recording'],
        ['Clinical Features', 'Age, Sex, Height, Weight, Nurse, Site, Device'],
        ['Disease Classes', 'NORM, MI, STTC, CD, HYP'],
        ['Label Source', 'SCP codes with diagnostic confidence scores'],
        ['Origin', 'Berlin, Germany — 1989 to 1996'],
    ]
    dt = Table(ds_data, colWidths=[5*cm, 10*cm])
    dt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY),
        ('TEXTCOLOR', (0,0), (-1,0), white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [white, LGRAY]),
        ('GRID', (0,0), (-1,-1), 0.5, HexColor('#d1d5db')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(dt)

    story.append(Paragraph("Disease Class Distribution", h2))
    story.append(Paragraph(
        "Labels are derived from SCP diagnostic codes. The dataset is imbalanced — "
        "NORM and CD are more common, while HYP is rare. Class-weighted loss was applied "
        "during training to compensate.", body))

    class_data = [
        ['Class', 'Full Name', 'Description', 'Loss Weight'],
        ['NORM', 'Normal', 'No cardiac abnormality detected', '0.45×'],
        ['MI',   'Myocardial Infarction', 'Heart attack — blocked coronary artery', '1.41×'],
        ['STTC', 'ST/T-wave Change', 'Ischaemia or repolarisation abnormality', '1.82×'],
        ['CD',   'Conduction Disorder', 'Bundle branch block, arrhythmia', '0.90×'],
        ['HYP',  'Hypertrophy', 'Enlarged heart chambers', '2.42×'],
    ]
    ct = Table(class_data, colWidths=[1.8*cm, 3.5*cm, 6.2*cm, 2.5*cm])
    ct.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY),
        ('TEXTCOLOR', (0,0), (-1,0), white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [white, LGRAY]),
        ('GRID', (0,0), (-1,-1), 0.5, HexColor('#d1d5db')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(ct)

    story.append(PageBreak())

    # ── SECTION 3: ARCHITECTURE ───────────────────────────────────────
    story.append(Paragraph("3. Model Architecture — CardioM3Net", h1))
    story.append(Paragraph(
        "CardioM3Net is a multimodal transformer-based network with 3,614,096 parameters. "
        "It has four main components that process ECG and clinical data independently "
        "before fusing them for joint prediction.", body))

    arch_data = [
        ['Component', 'Architecture', 'Output'],
        ['ECG Encoder',
         '4-block ResNet1D + Multi-head Self-Attention\n(Conv1D, BatchNorm, GELU, skip connections)',
         '256-dim embedding'],
        ['Clinical Encoder',
         '3-layer MLP with BatchNorm + Dropout\n(7 → 64 → 64)',
         '64-dim embedding'],
        ['Cross-Attention Fusion',
         '2-layer Transformer Encoder\nwith Modality Gate (learned α)',
         '256-dim fused repr.'],
        ['Task Heads',
         '3 independent FC heads (binary / disease / severity)',
         '2 / 5 / 3 class logits'],
        ['Domain Discriminator',
         'GRL + 2-layer MLP (3 domains)',
         'Domain probabilities'],
    ]
    at = Table(arch_data, colWidths=[3.5*cm, 7*cm, 3.5*cm])
    at.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY),
        ('TEXTCOLOR', (0,0), (-1,0), white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [white, LGRAY]),
        ('GRID', (0,0), (-1,-1), 0.5, HexColor('#d1d5db')),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(at)

    story.append(Paragraph("Architecture Hyperparameters", h2))
    hp_data = [
        ['Parameter', 'Value', 'Parameter', 'Value'],
        ['ECG Embed Dim', '256', 'Clinical Embed Dim', '64'],
        ['Fusion Dim', '256', 'Transformer Heads', '4'],
        ['Transformer Layers', '2', 'FF Dim', '512'],
        ['Dropout', '0.3', 'Disease Classes', '5'],
        ['Severity Bins', '3', 'Domain Count', '3'],
    ]
    ht = Table(hp_data, colWidths=[3.8*cm, 2.2*cm, 4.5*cm, 3.5*cm])
    ht.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), LBLUE),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [white, LGRAY]),
        ('GRID', (0,0), (-1,-1), 0.5, HexColor('#d1d5db')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(ht)

    # ── SECTION 4: TRAINING PIPELINE ─────────────────────────────────
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("4. Training Pipeline (5 Phases)", h1))
    story.append(Paragraph(
        "Training follows a sequential 5-phase pipeline. Each phase builds on the "
        "previous, progressively adding capabilities.", body))

    phases = [
        ("Phase 1", "Self-Supervised Pretraining (SimCLR)",
         "20 epochs, batch=64, lr=3e-4, temperature=0.07",
         "Trains the ECG encoder to learn robust ECG representations without labels "
         "using contrastive learning. Two augmented views of each ECG are created and "
         "the encoder learns to pull them together while pushing apart different ECGs. "
         "Final SimCLR loss: 0.0027."),
        ("Phase 2", "Multi-Task Supervised Training",
         "30 epochs (early stop), batch=32, lr=1e-3, patience=10",
         "Trains the full model end-to-end on all three tasks simultaneously. "
         "Class-weighted CrossEntropy loss compensates for disease imbalance. "
         "Gradient Reversal Layer enables domain adaptation across recording sites. "
         "Early stopping prevents overfitting."),
        ("Phase 3", "MAML Meta-Learning",
         "100 episodes, inner_steps=3, inner_lr=0.01, outer_lr=1e-4",
         "Model-Agnostic Meta-Learning trains the model to adapt to new cardiac "
         "conditions using only a few labelled examples (5-shot, 5-way). "
         "Gradient clipping (norm=1.0) prevents weight explosion during meta-updates."),
        ("Phase 4", "Domain Adaptation",
         "Built into Phase 2 — GRL lambda=0.1",
         "Gradient Reversal Layer (GRL) adversarially trains a domain discriminator "
         "to make ECG features domain-invariant across the 3 recording sites, "
         "improving generalisation to unseen hospitals."),
        ("Phase 5", "Federated Learning",
         "5 rounds, 3 clients, 2 local epochs per round",
         "Simulates privacy-preserving training across 3 hospitals using FedAvg "
         "aggregation. Each client trains on its local data partition; only model "
         "weights (not patient data) are shared and averaged."),
    ]

    for phase_id, name, config, desc in phases:
        pdata = [
            [Paragraph(f"<b>{phase_id}</b>", ParagraphStyle('pid', fontSize=11,
              textColor=white, fontName='Helvetica-Bold', alignment=TA_CENTER)),
             Paragraph(f"<b>{name}</b>", ParagraphStyle('pn', fontSize=11,
              textColor=NAVY, fontName='Helvetica-Bold')),
            ],
            ['', Paragraph(f"<i>{config}</i>", ParagraphStyle('pc', fontSize=8,
              textColor=GRAY, fontName='Helvetica-Oblique')),
            ],
            ['', Paragraph(desc, body)],
        ]
        pt = Table(pdata, colWidths=[2*cm, 12*cm])
        pt.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,-1), BLUE),
            ('BACKGROUND', (1,0), (1,0), LBLUE),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('GRID', (0,0), (-1,-1), 0.5, HexColor('#d1d5db')),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('SPAN', (0,0), (0,-1)),
        ]))
        story.append(KeepTogether([pt, Spacer(1, 0.2*cm)]))

    story.append(PageBreak())

    # ── SECTION 5: RESULTS ────────────────────────────────────────────
    story.append(Paragraph("5. Results", h1))
    story.append(Paragraph(
        "Training completed in 24.7 minutes on an NVIDIA GeForce RTX 3050 Laptop GPU (4 GB). "
        "Results below are on the held-out test set (4,087 records never seen during training).", body))

    story.append(Paragraph("Binary Classification (Normal vs Disease)", h2))
    res1 = [
        ['Metric', 'Central Model', 'Federated Model'],
        ['Accuracy',  '86.67%', '86.32%'],
        ['ROC-AUC',   '93.53%', '93.35%'],
        ['F1 Score',  '87.40%', '86.46%'],
        ['Precision', '~87%',   '~86%'],
        ['Recall',    '~88%',   '~87%'],
    ]
    rt1 = Table(res1, colWidths=[5*cm, 4.5*cm, 4.5*cm])
    rt1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY),
        ('TEXTCOLOR', (0,0), (-1,0), white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [white, LGREEN]),
        ('GRID', (0,0), (-1,-1), 0.5, HexColor('#d1d5db')),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(rt1)

    story.append(Paragraph("Multi-Class Classification", h2))
    res2 = [
        ['Task', 'Metric', 'Central Model', 'Federated Model'],
        ['Disease (5-class)', 'Accuracy',    '65.45%', '63.27%'],
        ['Disease (5-class)', 'F1 (weighted)','61.89%', '55.57%'],
        ['Severity (3-class)', 'Accuracy',   '83.73%', '83.07%'],
        ['Severity (3-class)', 'F1 (weighted)','81.63%', '~80%'],
    ]
    rt2 = Table(res2, colWidths=[4.5*cm, 3.5*cm, 3.5*cm, 3.5*cm])
    rt2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY),
        ('TEXTCOLOR', (0,0), (-1,0), white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [white, LGRAY]),
        ('GRID', (0,0), (-1,-1), 0.5, HexColor('#d1d5db')),
        ('ALIGN', (2,0), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(rt2)

    story.append(Paragraph("Result Analysis", h2))
    story.append(Paragraph(
        "The binary AUC of 93.5% is clinically significant — this means the model "
        "correctly ranks a cardiac patient above a healthy individual 93.5% of the time. "
        "The 5-class disease accuracy of 65.5% is in line with published results for "
        "PTB-XL 5-class classification using deep learning (literature range: 60–72%). "
        "Severity classification achieves 83.7%, which is strong given severity labels "
        "are derived indirectly from SCP confidence scores.", body))

    # ── SECTION 6: EXPLAINABILITY ─────────────────────────────────────
    story.append(Paragraph("6. Explainability", h1))
    story.append(Paragraph(
        "Two explainability methods were applied to make the model's decisions interpretable:", body))

    exp_items = [
        ("<b>Grad-CAM 1D (ECG Saliency)</b>",
         "Gradient-weighted Class Activation Mapping highlights which time regions "
         "and ECG leads influenced the disease prediction most. Saved as ecg_saliency.png."),
        ("<b>SHAP (Clinical Features)</b>",
         "SHapley Additive exPlanations using a Random Forest surrogate model shows "
         "which clinical features (age, sex, height, weight, site) contribute most. "
         "Saved as shap_clinical.png and shap_clinical_detail.png."),
        ("<b>Modality Weights</b>",
         "The learned fusion gate α shows the relative contribution of ECG vs clinical "
         "features per sample. ECG typically dominates (>80% weight). Saved as modality_weights.png."),
    ]
    for title, desc in exp_items:
        story.append(Paragraph(f"• {title}: {desc}", bullet))

    # Add result images if they exist
    result_images = [
        ('cardiom3net_results/roc_curves.png', 'ROC Curves — AUC per task'),
        ('cardiom3net_results/confusion_matrices.png', 'Confusion Matrices — per class breakdown'),
        ('cardiom3net_results/training_curves.png', 'Training Curves — Loss and Accuracy over epochs'),
    ]
    for img_path, cap in result_images:
        if os.path.exists(img_path):
            story.append(Spacer(1, 0.3*cm))
            try:
                img = Image(img_path, width=14*cm, height=7*cm, kind='proportional')
                story.append(img)
                story.append(Paragraph(cap, caption))
            except Exception:
                pass

    story.append(PageBreak())

    # ── SECTION 7: TECH STACK ─────────────────────────────────────────
    story.append(Paragraph("7. Technology Stack", h1))

    tech_data = [
        ['Category', 'Technology', 'Purpose'],
        ['Deep Learning', 'PyTorch 2.6', 'Model definition, training, GPU acceleration'],
        ['ECG Loading', 'wfdb', 'Reading PTB-XL .dat/.hea binary ECG files'],
        ['Data Processing', 'NumPy, Pandas', 'Array ops, CSV parsing, label extraction'],
        ['Preprocessing', 'scikit-learn', 'Imputer, StandardScaler, class weights, metrics'],
        ['Explainability', 'SHAP', 'Clinical feature importance via surrogate RF'],
        ['Visualisation', 'Matplotlib', 'Training curves, saliency maps, ROC, confusion'],
        ['Caching', 'NumPy .npz', 'Compressed cache of loaded ECG array (20K records)'],
        ['Progress', 'tqdm', 'ECG loading progress bar'],
        ['Frontend', 'React + TypeScript + Vite', 'Web application UI'],
        ['Backend', 'Node.js + Express', 'REST API, MongoDB, JWT auth'],
        ['GPU', 'NVIDIA RTX 3050 4GB', 'Training device (CUDA)'],
    ]
    tt = Table(tech_data, colWidths=[3.5*cm, 4*cm, 7.5*cm])
    tt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY),
        ('TEXTCOLOR', (0,0), (-1,0), white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [white, LGRAY]),
        ('GRID', (0,0), (-1,-1), 0.5, HexColor('#d1d5db')),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(tt)

    # ── SECTION 8: OUTPUT FILES ───────────────────────────────────────
    story.append(Paragraph("8. Output Files", h1))
    files = [
        ('cardiom3net_central.pth', '~14 MB', 'Main trained model weights + metadata'),
        ('cardiom3net_federated.pth', '~14 MB', 'Federated learning trained model'),
        ('clinical_imputer.pkl', '<1 KB', 'Fitted median imputer for clinical features'),
        ('clinical_scaler.pkl', '~1 KB', 'Fitted StandardScaler for clinical features'),
        ('training_curves.png', '224 KB', 'Loss and accuracy over training epochs'),
        ('roc_curves.png', '93 KB', 'ROC curves for all 3 tasks'),
        ('confusion_matrices.png', '120 KB', 'Confusion matrices per class'),
        ('ecg_saliency.png', '391 KB', 'Grad-CAM ECG saliency maps'),
        ('shap_clinical.png', '93 KB', 'SHAP beeswarm — clinical feature importance'),
        ('modality_weights.png', '31 KB', 'ECG vs clinical fusion gate weights'),
        ('_cache_*.npz', '~500 MB', 'Cached ECG array — enables fast re-runs'),
    ]
    fd = [['File', 'Size', 'Description']] + list(files)
    ft = Table(fd, colWidths=[5*cm, 1.8*cm, 8.2*cm])
    ft.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY),
        ('TEXTCOLOR', (0,0), (-1,0), white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [white, LGRAY]),
        ('GRID', (0,0), (-1,-1), 0.5, HexColor('#d1d5db')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(ft)

    # ── SECTION 9: LIMITATIONS & FUTURE WORK ─────────────────────────
    story.append(Paragraph("9. Limitations &amp; Future Work", h1))
    lims = [
        "Dataset is from a single country/era (Germany, 1989–1996) — may not generalise globally",
        "5-class accuracy (65.5%) leaves room for improvement; transformer pre-training on larger ECG corpora (e.g. MIMIC-IV-ECG) would help",
        "Clinical features are limited to 7 demographic/site variables — adding lab results (troponin, BNP) would improve severity prediction",
        "MAML query accuracy decreased during training — first-order MAML approximation needs more careful tuning",
        "No real federated infrastructure — federation is simulated on one machine",
        "Model not clinically validated — requires prospective trial before any clinical use",
    ]
    future = [
        "Download and fine-tune on CPSC 2018 or Georgia 12-lead for cross-dataset validation",
        "Replace first-order MAML with ANIL (Almost No Inner Loop) for more stable meta-learning",
        "Add 500 Hz ECG support (5,000 timesteps) for higher resolution analysis",
        "Integrate with the existing React web app via a Flask/FastAPI inference endpoint",
        "Add confidence calibration (temperature scaling) for better probability estimates",
    ]
    story.append(Paragraph("Limitations", h2))
    for lim in lims:
        story.append(Paragraph(f"• {lim}", bullet))
    story.append(Paragraph("Future Work", h2))
    for f in future:
        story.append(Paragraph(f"• {f}", bullet))

    # ── FOOTER ────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=LBLUE))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "CardioM3Net Project Report  ·  Generated April 2026  ·  "
        "GitHub: sharath2004-tech/cardiac-disease-detection",
        ParagraphStyle('footer', fontSize=8, textColor=GRAY,
                       fontName='Helvetica', alignment=TA_CENTER)))

    doc.build(story)
    print(f"PDF saved: {out}")
    import subprocess
    subprocess.Popen(['start', out], shell=True)

if __name__ == '__main__':
    build_pdf()
