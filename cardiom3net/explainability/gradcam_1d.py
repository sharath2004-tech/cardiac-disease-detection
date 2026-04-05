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

    # Plot saliency
    sal_time = np.linspace(0, ecg_sample.shape[1], len(saliency))
    axes[1].fill_between(sal_time, saliency, alpha=0.7, color='crimson')
    axes[1].set_title('Grad-CAM Saliency (importance)', fontweight='bold')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Importance')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'ecg_saliency.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
