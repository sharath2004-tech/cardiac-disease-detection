"""
MAML (Model-Agnostic Meta-Learning) training for fast adaptation.
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def maml_inner_loop(model, support_ecg, support_clin, support_labels,
                    inner_lr, inner_steps, device):
    """
    Perform inner-loop adaptation on the support set.
    Returns adapted model parameters (as a state_dict copy).
    """
    fast_model = copy.deepcopy(model)
    fast_optimizer = torch.optim.SGD(fast_model.parameters(), lr=inner_lr)
    criterion = nn.CrossEntropyLoss()

    fast_model.train()
    for _ in range(inner_steps):
        preds, _, _ = fast_model(support_ecg, support_clin)
        loss = criterion(preds['disease'], support_labels)
        fast_optimizer.zero_grad()
        loss.backward()
        fast_optimizer.step()

    return fast_model


def train_maml(model, task_dataset, config, device):
    """
    MAML meta-training over episodic tasks.
    Uses disease classification as the meta-learning objective.
    """
    print("\n" + "=" * 60)
    print("PHASE 3: MAML Meta-Learning")
    print("=" * 60)

    meta_optimizer = torch.optim.Adam(model.parameters(), lr=config.maml_outer_lr)
    criterion = nn.CrossEntropyLoss()

    meta_losses = []

    for episode in range(1, config.maml_episodes + 1):
        episode_data = task_dataset.sample_episode()

        support_ecg = episode_data['support_ecg'].to(device)
        support_clin = episode_data['support_clinical'].to(device)
        support_labels = episode_data['support_labels'].to(device)

        query_ecg = episode_data['query_ecg'].to(device)
        query_clin = episode_data['query_clinical'].to(device)
        query_labels = episode_data['query_labels'].to(device)

        # Inner loop: adapt on support set
        adapted_model = maml_inner_loop(
            model, support_ecg, support_clin, support_labels,
            config.maml_inner_lr, config.maml_inner_steps, device
        )

        # Outer loop: evaluate on query set, backprop through original model
        adapted_model.eval()
        query_preds, _, _ = adapted_model(query_ecg, query_clin)
        query_loss = criterion(query_preds['disease'], query_labels)

        # Approximate first-order MAML: update original model toward adapted model
        meta_optimizer.zero_grad()

        # Copy gradients from adapted model to original model
        with torch.no_grad():
            for orig_param, adapted_param in zip(model.parameters(), adapted_model.parameters()):
                if orig_param.requires_grad:
                    diff = orig_param.data - adapted_param.data
                    orig_param.grad = diff * config.maml_outer_lr

        meta_optimizer.step()
        meta_losses.append(query_loss.item())

        if episode % 50 == 0 or episode == 1:
            avg_loss = np.mean(meta_losses[-50:])
            query_acc = (query_preds['disease'].argmax(1) == query_labels).float().mean().item()
            print(f"  Episode {episode:4d}/{config.maml_episodes} | "
                  f"Query Loss: {avg_loss:.4f} | Query Acc: {query_acc:.3f}")

    print(f"  MAML training complete. Final avg loss: {np.mean(meta_losses[-50:]):.4f}")
    return model, meta_losses
