"""
1D Grad-CAM for ECG saliency visualization.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


class GradCAM1D:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self._fh = target_layer.register_forward_hook(self._save_act)
        self._bh = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, _m, _i, output):
        self.activations = output.detach()

    def _save_grad(self, _m, _gi, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, ecg_tensor, clinical_tensor):
        self.model.eval()
        self.model.zero_grad()
        preds, _, _ = self.model(ecg_tensor, clinical_tensor)
        logits = preds['binary']
        logits[:, 1].sum().backward()

        weights = self.gradients.mean(dim=2, keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1)).squeeze(0)
        cam = cam.cpu().numpy()
        cam = cam / (cam.max() + 1e-6)
        return cam

    def upsample(self, cam, target_length):
        """Linearly interpolate saliency map to the full ECG signal length."""
        from scipy.interpolate import interp1d
        x_old = np.linspace(0, 1, len(cam))
        x_new = np.linspace(0, 1, target_length)
        return interp1d(x_old, cam, kind='linear')(x_new)

    def close(self):
        self._fh.remove()
        self._bh.remove()


def plot_ecg_saliency(model, ecg_sample, clinical_sample, device, output_dir, sample_idx=0):
    """Generate and save Grad-CAM saliency map for one ECG sample."""
    os.makedirs(output_dir, exist_ok=True)

    ecg_t = torch.tensor(ecg_sample[None, ...], dtype=torch.float32, device=device)
    clin_t = torch.tensor(clinical_sample[None, ...], dtype=torch.float32, device=device)

    # Target: last conv layer of ECG encoder
    target_layer = model.ecg_encoder.layer3[-1].conv2
    gradcam = GradCAM1D(model, target_layer)
    saliency = gradcam.generate(ecg_t, clin_t)
    gradcam.close()

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), gridspec_kw={'height_ratios': [3, 1]})

    # Plot ECG leads (first 3)
    time = np.arange(ecg_sample.shape[1])
    for lead_idx in range(min(3, ecg_sample.shape[0])):
        axes[0].plot(time, ecg_sample[lead_idx], alpha=0.7, label=f'Lead {lead_idx + 1}')
    axes[0].set_title(f'ECG Signal (Sample {sample_idx})', fontweight='bold')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Plot saliency — upsample from conv output resolution to full signal length
    signal_length = ecg_sample.shape[1]
    saliency_full = gradcam.upsample(saliency, signal_length)
    time = np.arange(signal_length)
    axes[1].fill_between(time, saliency_full, alpha=0.7, color='crimson')
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title('Grad-CAM Saliency (importance)', fontweight='bold')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Importance')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'ecg_saliency.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_multi_sample_saliency(model, ecg_samples, clinical_samples, disease_labels,
                               class_names, device, output_dir):
    """Grad-CAM saliency grid: one ECG sample per disease class side-by-side with
    a saliency overlay and per-lead mean importance bar chart."""
    os.makedirs(output_dir, exist_ok=True)

    # Select one representative sample per class
    disease_labels = np.asarray(disease_labels, dtype=int)
    selected_idx, selected_cls = [], []
    for c in range(len(class_names)):
        cls_idx = np.where(disease_labels == c)[0]
        if len(cls_idx) > 0:
            selected_idx.append(int(cls_idx[0]))
            selected_cls.append(c)

    n = len(selected_idx)
    if n == 0:
        return

    target_layer = model.ecg_encoder.layer3[-1].conv2
    gradcam = GradCAM1D(model, target_layer)

    fig, axes = plt.subplots(n, 2, figsize=(16, 3.5 * n),
                              gridspec_kw={'width_ratios': [4, 1]})
    if n == 1:
        axes = axes[None, :]

    colors_lead = ['#1976D2', '#43A047', '#FB8C00']

    for row, (idx, cls) in enumerate(zip(selected_idx, selected_cls)):
        ecg_s = ecg_samples[idx]
        clin_s = clinical_samples[idx]

        ecg_t = torch.tensor(ecg_s[None, ...], dtype=torch.float32, device=device)
        clin_t = torch.tensor(clin_s[None, ...], dtype=torch.float32, device=device)

        saliency_raw = gradcam.generate(ecg_t, clin_t)
        signal_length = ecg_s.shape[1]
        saliency_full = gradcam.upsample(saliency_raw, signal_length)
        time = np.arange(signal_length)

        ax_ecg = axes[row, 0]
        n_leads_shown = min(3, ecg_s.shape[0])
        for lead in range(n_leads_shown):
            ax_ecg.plot(time, ecg_s[lead], alpha=0.65, linewidth=0.9,
                        color=colors_lead[lead], label=f'Lead {lead + 1}')
        ax2 = ax_ecg.twinx()
        ax2.fill_between(time, saliency_full, alpha=0.28, color='crimson')
        ax2.set_ylim(0, 1.5)
        ax2.set_ylabel('Saliency', fontsize=8, color='crimson')
        ax2.tick_params(axis='y', colors='crimson', labelsize=7)
        ax_ecg.set_title(f'Class: {class_names[cls]}', fontweight='bold')
        ax_ecg.set_ylabel('Amplitude')
        ax_ecg.legend(loc='upper right', fontsize=7)
        ax_ecg.grid(True, alpha=0.2)

        # Per-lead mean saliency bar (approximated: same saliency map per lead)
        ax_bar = axes[row, 1]
        lead_saliency = []
        for lead in range(n_leads_shown):
            # Weight saliency by absolute signal amplitude of each lead
            lead_weight = np.abs(ecg_s[lead]) / (np.abs(ecg_s[:n_leads_shown]).sum(axis=0) + 1e-8)
            lead_saliency.append(float((saliency_full * lead_weight).mean()))
        ax_bar.barh(range(n_leads_shown), lead_saliency,
                    color=colors_lead[:n_leads_shown], edgecolor='white')
        ax_bar.set_yticks(range(n_leads_shown))
        ax_bar.set_yticklabels([f'Lead {i + 1}' for i in range(n_leads_shown)])
        ax_bar.set_xlabel('Weighted Saliency')
        ax_bar.set_title('Lead Importance', fontweight='bold')
        ax_bar.grid(True, alpha=0.3, axis='x')

    gradcam.close()
    plt.suptitle('Grad-CAM ECG Saliency — One Sample per Disease Class',
                 fontweight='bold', fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, 'ecg_saliency_grid.png')
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
