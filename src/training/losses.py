import torch
import torch.nn as nn
import numpy as np

from sklearn.utils.class_weight import compute_class_weight


# for weighted CrossEntropyLoss
def get_age_weights(age_labels, num_classes, device='cpu'):
    classes = np.arange(num_classes)
    weights = compute_class_weight(
        'balanced', 
        classes=classes, 
        y=age_labels
    )
    return torch.tensor(weights, dtype=torch.float32).to(device)


def get_loss_function(age_weights=None):
    return {
        'gender': nn.BCEWithLogitsLoss(),
        'age'   : nn.CrossEntropyLoss(weight=age_weights),
        'race'  : nn.CrossEntropyLoss()   
    }


def compute_losses(pred, target, loss_funcs, loss_weights=None):
    losses = {}

     # gender
    losses["gender"] = loss_funcs["gender"](
        pred["gender"].squeeze(1),
        target["gender"].float()
    )

    # age
    losses["age"] = loss_funcs["age"](
        pred["age"],
        target["age"]
    )

    # race
    losses["race"] = loss_funcs["race"](
        pred["race"],
        target["race"]
    )


    if loss_weights is None:
        loss_weights = {
            "gender": 1.0,
            "age": 1.0,
            "race": 1.0
        }


    total_loss = (
        loss_weights["gender"] * losses["gender"]
        +
        loss_weights["age"] * losses["age"]
        +
        loss_weights["race"] * losses["race"]
    )


    return total_loss, losses
