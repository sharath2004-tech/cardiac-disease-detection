"""
Federated Learning with Weighted FedAvg.
Simulates cross-silo hospital training.
"""
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .supervised import run_epoch, MultiTaskLoss


def weighted_fedavg(state_dicts, weights):
    total = float(sum(weights))
    averaged = {}
    for key in state_dicts[0]:
        acc = None
        for sd, w in zip(state_dicts, weights):
            val = sd[key].detach().clone().float() * (w / total)
            acc = val if acc is None else acc + val
        averaged[key] = acc.type_as(state_dicts[0][key])
    return averaged


def run_federated(model, train_dataset, val_loader, config, device):
    """
    Simulate federated learning across multiple clients.
    """
    print("\n" + "=" * 60)
    print(f"PHASE 5: Federated Learning ({config.fed_clients} clients, {config.fed_rounds} rounds)")
    print("=" * 60)

    criterion = MultiTaskLoss(
        lambda_binary=config.lambda_binary,
        lambda_disease=config.lambda_disease,
        lambda_severity=config.lambda_severity,
    )

    # Split training data among clients
    indices = np.random.permutation(len(train_dataset))
    splits = np.array_split(indices, config.fed_clients)
    client_subsets = [Subset(train_dataset, s.tolist()) for s in splits if len(s) > 0]

    fed_history = []

    for rnd in range(1, config.fed_rounds + 1):
        local_states, local_weights = [], []

        for cid, subset in enumerate(client_subsets, 1):
            local_model = copy.deepcopy(model)
            local_opt = torch.optim.Adam(local_model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
            loader = DataLoader(subset, batch_size=config.batch_size, shuffle=True)

            for _ in range(config.fed_local_epochs):
                run_epoch(local_model, loader, criterion, device, local_opt)

            local_states.append(copy.deepcopy(local_model.state_dict()))
            local_weights.append(len(subset))

        # Aggregate
        global_state = weighted_fedavg(local_states, local_weights)
        model.load_state_dict(global_state)

        # Evaluate
        val_res = run_epoch(model, val_loader, criterion, device)
        fed_history.append(val_res)

        print(f"  Round {rnd}/{config.fed_rounds} | "
              f"Bin-Acc: {val_res['binary']['accuracy']:.3f} "
              f"Bin-AUC: {val_res['binary']['roc_auc']:.3f} | "
              f"Dis-Acc: {val_res['disease']['accuracy']:.3f}")

    # Final evaluation
    final = run_epoch(model, val_loader, criterion, device)
    print(f"\n  Federated final:")
    print(f"    Binary  — Acc: {final['binary']['accuracy']:.4f} AUC: {final['binary']['roc_auc']:.4f}")
    print(f"    Disease — Acc: {final['disease']['accuracy']:.4f}")

    return model, fed_history, final
