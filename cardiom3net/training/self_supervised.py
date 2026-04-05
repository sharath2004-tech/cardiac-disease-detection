"""
SimCLR-style self-supervised pretraining for the ECG encoder.
Learns robust ECG representations before supervised fine-tuning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..models.ecg_encoder import ECGEncoder
from ..models.cardiom3net import SimCLRHead
from ..data.multimodal_dataset import SimCLRDataset


def nt_xent_loss(z1, z2, temperature=0.07):
    """Normalized Temperature-scaled Cross Entropy Loss."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    B = z1.shape[0]

    z = torch.cat([z1, z2], dim=0)                      # (2B, D)
    sim = torch.mm(z, z.t()) / temperature               # (2B, 2B)

    # Mask out self-similarity
    mask = ~torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_select(mask).view(2 * B, -1)        # (2B, 2B-1)

    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([
        torch.arange(B - 1, 2 * B - 1),
        torch.arange(0, B),
    ]).to(z.device)

    return F.cross_entropy(sim, labels)


def pretrain_ecg_encoder(ecg_array, config, device):
    """
    Run SimCLR pretraining on raw ECG signals.
    Returns a pretrained ECGEncoder with learned weights.
    """
    print("\n" + "=" * 60)
    print("PHASE 1: Self-Supervised ECG Pretraining (SimCLR)")
    print("=" * 60)

    encoder = ECGEncoder(
        num_leads=config.num_leads,
        embed_dim=config.ecg_embed_dim,
    ).to(device)

    projection = SimCLRHead(
        input_dim=config.ecg_embed_dim,
        output_dim=config.ssl_projection_dim,
    ).to(device)

    dataset = SimCLRDataset(ecg_array)
    loader = DataLoader(dataset, batch_size=config.ssl_batch_size,
                        shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projection.parameters()),
        lr=config.ssl_lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.ssl_epochs)

    for epoch in range(1, config.ssl_epochs + 1):
        encoder.train()
        projection.train()
        total_loss = 0.0
        n_batches = 0

        for view1, view2 in loader:
            view1, view2 = view1.to(device), view2.to(device)

            h1 = encoder(view1)
            h2 = encoder(view2)
            z1 = projection(h1)
            z2 = projection(h2)

            loss = nt_xent_loss(z1, z2, config.ssl_temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{config.ssl_epochs} | SimCLR Loss: {avg_loss:.4f}")

    print(f"  SSL pretraining complete. Final loss: {avg_loss:.4f}")
    return encoder
