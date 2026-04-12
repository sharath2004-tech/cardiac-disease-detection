"""
Generate two publication-quality architecture diagrams for CardioM3Net.
Accurately reflects the actual workspace implementation.

Run: python generate_diagrams.py
Outputs: cardiom3net_results/diagram1_architecture.png
         cardiom3net_results/diagram2_pipeline.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

os.makedirs("cardiom3net_results", exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
C = {
    "input":      "#1A237E",   # deep navy
    "pcg":        "#4A148C",   # deep purple
    "ecg":        "#1565C0",   # deep blue
    "clinical":   "#BF360C",   # deep orange
    "fusion":     "#1B5E20",   # deep green
    "tasks":      "#E65100",   # amber-orange
    "xai":        "#880E4F",   # deep pink
    "ssl":        "#006064",   # teal
    "maml":       "#4E342E",   # brown
    "fed":        "#37474F",   # blue-grey
    "da":         "#558B2F",   # light green dark
    "white":      "#FFFFFF",
    "bg":         "#F8F9FA",
    "arrow":      "#424242",
}

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def box(ax, cx, cy, w, h, label, sublabel=None,
        fc="#1565C0", ec="white", fontsize=9, radius=0.04,
        text_color="white", bold=True):
    """Draw a rounded rectangle with centred label."""
    rect = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0.01,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=1.4, zorder=3,
    )
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    if sublabel:
        ax.text(cx, cy + h * 0.12, label, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight=weight, zorder=4)
        ax.text(cx, cy - h * 0.22, sublabel, ha="center", va="center",
                fontsize=fontsize - 1.5, color=text_color, alpha=0.85, zorder=4,
                style="italic")
    else:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight=weight, zorder=4,
                wrap=True)


def arrow(ax, x0, y0, x1, y1, color="#424242", lw=1.4, style="->"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"),
                zorder=2)


def curved_arrow(ax, x0, y0, x1, y1, rad=0.25, color="#424242", lw=1.4):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=2)


def section_label(ax, x, y, text, color="#37474F", fontsize=11):
    ax.text(x, y, text, ha="left", va="center",
            fontsize=fontsize, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#ECEFF1", ec=color, lw=1.2))


# ═════════════════════════════════════════════════════════════════════════════
#  DIAGRAM 1 — Full Architecture (top-down data flow)
# ═════════════════════════════════════════════════════════════════════════════

fig1, ax1 = plt.subplots(figsize=(14, 11))
ax1.set_xlim(0, 14)
ax1.set_ylim(0, 11)
ax1.axis("off")
fig1.patch.set_facecolor(C["bg"])
ax1.set_facecolor(C["bg"])

# Title
ax1.text(7, 10.55, "CardioM3Net: Multimodal Meta-Learning Framework",
         ha="center", va="center", fontsize=15, fontweight="bold", color="#1A237E")
ax1.text(7, 10.18, "for Cardiovascular Disease Diagnosis",
         ha="center", va="center", fontsize=12, color="#37474F")

# ── Row 0: Inputs ─────────────────────────────────────────────────────────────
box(ax1, 1.2, 9.2, 1.7, 0.62, "PCG Signals",   "CinC 2016",    fc=C["pcg"])
box(ax1, 1.2, 8.3, 1.7, 0.62, "ECG Signals",   "PTB-XL",       fc=C["ecg"])
box(ax1, 1.2, 7.4, 1.7, 0.62, "Clinical Data", "13 features",  fc=C["clinical"])

# ── Row 1: Encoders ───────────────────────────────────────────────────────────
box(ax1, 4.2, 9.2, 2.5, 0.62,
    "PCG Encoder",
    "2D CNN · Log-Mel Spectrogram\n(B,1,64,99) → 128-dim",
    fc=C["pcg"], fontsize=8)

box(ax1, 4.2, 8.3, 2.5, 0.62,
    "ECG Encoder",
    "ResNet1D + Self-Attention\n(B,12,T) → 256-dim",
    fc=C["ecg"], fontsize=8)

box(ax1, 4.2, 7.4, 2.5, 0.62,
    "Clinical Encoder",
    "MLP [128→64→64-dim]\nBatchNorm + Dropout",
    fc=C["clinical"], fontsize=8)

# Input → Encoder arrows
for y in [9.2, 8.3, 7.4]:
    arrow(ax1, 2.05, y, 2.95, y)

# ── Row 1b: SSL over ECG ──────────────────────────────────────────────────────
box(ax1, 4.2, 6.35, 2.5, 0.58,
    "SimCLR Pretraining",
    "NT-Xent Loss · ECG augmentation\n(Phase 1 — offline)",
    fc=C["ssl"], fontsize=8)
curved_arrow(ax1, 3.5, 8.0, 3.5, 6.65, rad=-0.25, color=C["ssl"])

# ── Row 2: Fusion ─────────────────────────────────────────────────────────────
box(ax1, 7.9, 8.3, 2.8, 1.5,
    "CrossAttention Fusion\n+ Modality Gate",
    "Gate floor=0.20 · Diversity loss λ=0.1\n"
    "3 tokens → Transformer (4h×2L)\n"
    "→ fused 256-dim vector",
    fc=C["fusion"], fontsize=8, radius=0.05)

# Encoder → Fusion arrows
arrow(ax1, 5.45, 9.2,  6.5,  8.8)
arrow(ax1, 5.45, 8.3,  6.5,  8.3)
arrow(ax1, 5.45, 7.4,  6.5,  7.8)

# ── Row 3: Task Heads ─────────────────────────────────────────────────────────
box(ax1, 11.2, 9.3,  2.4, 0.55, "Binary Head",    "Normal / Abnormal",       fc=C["tasks"], fontsize=8)
box(ax1, 11.2, 8.3,  2.4, 0.55, "Disease Head",   "5-class (NORM/MI/STTC/CD/HYP)", fc=C["tasks"], fontsize=8)
box(ax1, 11.2, 7.3,  2.4, 0.55, "Severity Head",  "3-class (mild/mod/severe)", fc=C["tasks"], fontsize=8)

# Fusion → Head arrows
arrow(ax1, 9.3, 8.75, 9.9, 9.3)
arrow(ax1, 9.3, 8.3,  9.9, 8.3)
arrow(ax1, 9.3, 7.85, 9.9, 7.3)

# ── Domain Adaptation ─────────────────────────────────────────────────────────
box(ax1, 7.9, 6.35, 2.8, 0.58,
    "Domain Discriminator",
    "GRL · Adversarial · 3 domains (devices)\nPhase 2 — simultaneous",
    fc=C["da"], fontsize=8)

curved_arrow(ax1, 5.3, 8.3, 6.5, 6.55, rad=0.3, color=C["da"], lw=1.2)
arrow(ax1, 9.3, 6.35, 9.9, 7.1, color=C["da"])

# ── MAML ──────────────────────────────────────────────────────────────────────
box(ax1, 4.2, 5.15, 2.5, 0.58,
    "MAML Meta-Learning",
    "Fast adaptation · 5-way 5-shot\nInner SGD + outer Adam (Phase 3)",
    fc=C["maml"], fontsize=8)

# ── Federated Learning ────────────────────────────────────────────────────────
box(ax1, 7.9, 5.15, 2.8, 0.58,
    "Federated Learning",
    "Weighted FedAvg · 3 hospital clients\n5 rounds × 2 local epochs (Phase 4)",
    fc=C["fed"], fontsize=8)

arrow(ax1, 5.45, 5.15, 6.5, 5.15)
arrow(ax1, 9.3,  5.15, 9.9, 6.35, color=C["fed"], lw=1.2)

curved_arrow(ax1, 3.5, 7.69, 3.5, 5.45, rad=0.25, color=C["maml"])

# ── Explainability ─────────────────────────────────────────────────────────────
box(ax1, 7.9, 4.05, 2.8, 0.58,
    "Explainable AI (Phase 5)",
    "SHAP clinical importance\nECG saliency · Modality weight plot",
    fc=C["xai"], fontsize=8)

arrow(ax1, 7.9, 4.85, 7.9, 4.35)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(fc=C["pcg"],      label="PCG Branch"),
    mpatches.Patch(fc=C["ecg"],      label="ECG Branch"),
    mpatches.Patch(fc=C["clinical"], label="Clinical Branch"),
    mpatches.Patch(fc=C["fusion"],   label="Fusion / Attention"),
    mpatches.Patch(fc=C["tasks"],    label="Multi-Task Heads"),
    mpatches.Patch(fc=C["ssl"],      label="Self-Supervised (SimCLR)"),
    mpatches.Patch(fc=C["da"],       label="Domain Adaptation (GRL)"),
    mpatches.Patch(fc=C["maml"],     label="MAML Meta-Learning"),
    mpatches.Patch(fc=C["fed"],      label="Federated Learning"),
    mpatches.Patch(fc=C["xai"],      label="Explainable AI"),
]
ax1.legend(handles=legend_items, loc="lower left", fontsize=7.5,
           ncol=2, framealpha=0.92, edgecolor="#90A4AE",
           bbox_to_anchor=(0.01, 0.01))

# Bounding box around full diagram
rect = FancyBboxPatch((0.1, 3.6), 13.8, 7.2,
                      boxstyle="round,pad=0.05", fill=False,
                      edgecolor="#90A4AE", linewidth=1.5, zorder=0)
ax1.add_patch(rect)

fig1.tight_layout(pad=0.3)
out1 = "cardiom3net_results/diagram1_architecture.png"
fig1.savefig(out1, dpi=180, bbox_inches="tight", facecolor=C["bg"])
plt.close(fig1)
print(f"Saved: {out1}")


# ═════════════════════════════════════════════════════════════════════════════
#  DIAGRAM 2 — Multi-panel: Pipeline + Deployment + Novelty + Detailed
# ═════════════════════════════════════════════════════════════════════════════

fig2 = plt.figure(figsize=(16, 14))
fig2.patch.set_facecolor(C["bg"])

# ── Panel layout: 2×2 + bottom strip ──────────────────────────────────────────
ax_a = fig2.add_axes([0.02, 0.54, 0.46, 0.40])   # (a) Training Pipeline
ax_b = fig2.add_axes([0.52, 0.54, 0.46, 0.40])   # (b) Deployment Flow
ax_c = fig2.add_axes([0.02, 0.10, 0.46, 0.40])   # (c) Key Novelty & Impact
ax_d = fig2.add_axes([0.52, 0.10, 0.46, 0.40])   # (d) Detailed Architecture

fig2.text(0.5, 0.975, "CardioM3Net Framework — Training, Deployment & Architecture",
          ha="center", va="center", fontsize=14, fontweight="bold", color="#1A237E")

# ─── (a) Training Pipeline ────────────────────────────────────────────────────
ax_a.set_xlim(0, 10); ax_a.set_ylim(0, 10); ax_a.axis("off")
ax_a.set_facecolor(C["bg"])
ax_a.text(5, 9.5, "(a)  5-Phase Training Pipeline",
          ha="center", fontsize=11, fontweight="bold", color="#1A237E")

phases = [
    (5, 8.2, "Phase 1: SimCLR Pretraining",    "ECG contrastive learning\nNT-Xent loss · augmentation pairs",            C["ssl"]),
    (5, 6.7, "Phase 2: Supervised Multi-Task",  "Binary + Disease + Severity heads\nDomain Adaptation (GRL) · λ=0.1",      C["ecg"]),
    (5, 5.2, "Phase 3: MAML Meta-Learning",     "5-way 5-shot · inner SGD 3 steps\nOuter Adam lr=1e-4 · 100 episodes",      C["maml"]),
    (5, 3.7, "Phase 4: Federated Learning",     "Weighted FedAvg · 3 clients\n5 rounds × 2 local epochs",                  C["fed"]),
    (5, 2.2, "Phase 5: Explainability",         "SHAP · ECG saliency\nModality gate weights visualization",                C["xai"]),
]

for cx, cy, title, sub, fc in phases:
    box(ax_a, cx, cy, 8.5, 1.0, title, sub, fc=fc, fontsize=9)

for i in range(len(phases) - 1):
    arrow(ax_a, 5, phases[i][1] - 0.5, 5, phases[i+1][1] + 0.5)

# ─── (b) Deployment Flow ──────────────────────────────────────────────────────
ax_b.set_xlim(0, 10); ax_b.set_ylim(0, 10); ax_b.axis("off")
ax_b.set_facecolor(C["bg"])
ax_b.text(5, 9.5, "(b)  Deployment / Inference Flow",
          ha="center", fontsize=11, fontweight="bold", color="#1A237E")

deploy_steps = [
    (5, 8.4,  "New Patient Data",                 "ECG + Clinical (PCG optional)",               C["input"]),
    (5, 7.05, "Preprocessing",                    "Median impute · StandardScaler\nLog-mel spec", C["ecg"]),
    (5, 5.7,  "Encoder Forward Pass",             "ECG 256-d · Clinical 64-d · PCG 128-d",       C["fusion"]),
    (5, 4.35, "CrossAttention Fusion + Gate",     "ModalityGate (floor 0.20)\n256-d fused vector", C["fusion"]),
    (5, 3.0,  "Multi-Task Prediction",            "Normal/Abnormal · Disease class · Severity",  C["tasks"]),
    (5, 1.65, "Explainable Report",               "SHAP importance · Saliency map\nModality weights", C["xai"]),
]

for cx, cy, title, sub, fc in deploy_steps:
    box(ax_b, cx, cy, 8.5, 0.95, title, sub, fc=fc, fontsize=9)

for i in range(len(deploy_steps) - 1):
    arrow(ax_b, 5, deploy_steps[i][1] - 0.48, 5, deploy_steps[i+1][1] + 0.48)

# ─── (c) Key Novelty & Impact ─────────────────────────────────────────────────
ax_c.set_xlim(0, 10); ax_c.set_ylim(0, 10); ax_c.axis("off")
ax_c.set_facecolor(C["bg"])
ax_c.text(5, 9.5, "(c)  Key Novelty & Contributions",
          ha="center", fontsize=11, fontweight="bold", color="#1A237E")

novelties = [
    (5, 8.3,  "Trimodal Fusion",
              "First fusion of ECG + PCG + Clinical\nwith learned modality gating (floor + entropy reg.)", C["fusion"]),
    (5, 6.8,  "SimCLR ECG Pretraining",
              "Contrastive self-supervised ECG encoder\npretrained before supervised fine-tuning",           C["ssl"]),
    (5, 5.3,  "MAML for Cardiac Adaptation",
              "Fast cross-dataset generalisation\n5-way 5-shot disease classification",                     C["maml"]),
    (5, 3.8,  "Privacy-Preserving Federated Learning",
              "Weighted FedAvg across 3 hospital silos\nNo raw patient data sharing",                       C["fed"]),
    (5, 2.3,  "Clinically Interpretable XAI",
              "SHAP feature importance + ResNet1D saliency\n+ per-modality contribution visualization",     C["xai"]),
]

for cx, cy, title, sub, fc in novelties:
    box(ax_c, cx, cy, 8.5, 1.0, title, sub, fc=fc, fontsize=9)

# ─── (d) Detailed Architecture ────────────────────────────────────────────────
ax_d.set_xlim(0, 10); ax_d.set_ylim(0, 10); ax_d.axis("off")
ax_d.set_facecolor(C["bg"])
ax_d.text(5, 9.5, "(d)  CardioM3Net: Detailed Architecture",
          ha="center", fontsize=11, fontweight="bold", color="#1A237E")

# Inputs column
box(ax_d, 1.1, 8.3,  1.7, 0.62, "PCG",       "CinC 2016",   fc=C["pcg"],      fontsize=8)
box(ax_d, 1.1, 7.2,  1.7, 0.62, "ECG",       "PTB-XL",      fc=C["ecg"],      fontsize=8)
box(ax_d, 1.1, 6.1,  1.7, 0.62, "Clinical",  "13 features", fc=C["clinical"], fontsize=8)

# Encoder column
box(ax_d, 3.7, 8.3, 2.6, 0.62, "PCG Encoder",      "2D CNN · Log-Mel\n128-dim",           fc=C["pcg"],      fontsize=8)
box(ax_d, 3.7, 7.2, 2.6, 0.62, "ECG Encoder",      "ResNet1D+Attention\n256-dim",          fc=C["ecg"],      fontsize=8)
box(ax_d, 3.7, 6.1, 2.6, 0.62, "Clinical Encoder", "MLP BN 128→64\n64-dim",               fc=C["clinical"], fontsize=8)

# SSL under ECG encoder
box(ax_d, 3.7, 5.0, 2.6, 0.55, "SimCLR Pretrain",  "Phase 1 (offline)\nNT-Xent loss",     fc=C["ssl"],      fontsize=8)

for y in [8.3, 7.2, 6.1]:
    arrow(ax_d, 1.95, y, 2.40, y)

curved_arrow(ax_d, 3.2, 6.9, 3.2, 5.28, rad=0.2, color=C["ssl"])

# Fusion
box(ax_d, 6.8, 7.2, 2.4, 1.55,
    "CrossAttention\nFusion + Gate",
    "Gate floor 0.20\nDiv. loss λ=0.1\n4-head × 2-layer\nTransformer\n→ 256-dim",
    fc=C["fusion"], fontsize=8, radius=0.05)

arrow(ax_d, 5.0, 8.3, 5.6, 7.8)
arrow(ax_d, 5.0, 7.2, 5.6, 7.2)
arrow(ax_d, 5.0, 6.1, 5.6, 6.6)

# Domain Adaptation feeds from ECG encoder
box(ax_d, 6.8, 5.0, 2.4, 0.55,
    "Domain Discriminator",
    "GRL · 3 device domains\nPhase 2 adversarial",
    fc=C["da"], fontsize=8)

curved_arrow(ax_d, 5.0, 7.2, 5.6, 5.28, rad=0.3, color=C["da"], lw=1.0)

# Output heads
box(ax_d, 9.4, 8.4, 1.7, 0.52, "Binary",    "Normal/Abnorm.", fc=C["tasks"], fontsize=8)
box(ax_d, 9.4, 7.2, 1.7, 0.52, "Disease",   "5-class",        fc=C["tasks"], fontsize=8)
box(ax_d, 9.4, 6.0, 1.7, 0.52, "Severity",  "3-class",        fc=C["tasks"], fontsize=8)

arrow(ax_d, 8.0, 7.75, 8.5, 8.4)
arrow(ax_d, 8.0, 7.2,  8.5, 7.2)
arrow(ax_d, 8.0, 6.65, 8.5, 6.0)

# XAI at bottom
box(ax_d, 5.0, 3.7, 7.5, 0.75,
    "Explainable AI (Phase 5)",
    "SHAP clinical importance · ECG ResNet1D saliency · Modality gate weight visualization",
    fc=C["xai"], fontsize=8)

arrow(ax_d, 5.0, 4.73, 5.0, 4.08)

# Panel borders
for ax, label in [(ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d")]:
    for spine in ax.spines.values():
        spine.set_visible(False)

fig2.tight_layout(pad=0.3)
out2 = "cardiom3net_results/diagram2_pipeline.png"
fig2.savefig(out2, dpi=180, bbox_inches="tight", facecolor=C["bg"])
plt.close(fig2)
print(f"Saved: {out2}")
print("Done.")
