import torch
import torch.nn as nn
import numpy as np

from sklearn.utils.class_weight import compute_class_weight
from .age_strategy import get_age_strategy

DEFAULT_LOSS_WEIGHTS = {
    "gender": 1.0, "age": 1.0, "race": 1.0
}
AGE_MAE_WEIGHT = 0.1


# for weighted CrossEntropyLoss
def get_age_weights(age_labels, num_classes, device='cpu'):
    classes = np.arange(num_classes)
    weights = compute_class_weight(
        'balanced', 
        classes=classes, 
        y=age_labels
    )
    weights = np.sqrt(weights)

    return torch.tensor(weights, dtype=torch.float32).to(device)


# for weighted CrossEntropyLoss
def get_race_weights(race_labels, num_classes, device='cpu'):
    classes = np.arange(num_classes) 
    weights = compute_class_weight(
         'balanced', 
         classes=classes, 
         y=race_labels
    ) 
     
    return torch.tensor(weights, dtype=torch.float32).to(device)



def get_loss_function(age_loss_type="ce", age_weights=None, race_weights=None, num_age_classes=9):
    return {
        "gender":       nn.BCEWithLogitsLoss(),
        "race":         nn.CrossEntropyLoss(weight=race_weights),
        "age_strategy": get_age_strategy(age_loss_type, age_weights=age_weights, num_age_classes=num_age_classes),
    }


def compute_losses(pred, target, loss_funcs, age_mae_weight=AGE_MAE_WEIGHT, loss_weights=None):
    age_strategy = loss_funcs["age_strategy"]

    losses = {
        "gender": loss_funcs["gender"]
            (pred["gender"].squeeze(1), target["gender"].float()
        ),
        "race":   loss_funcs["race"](
            pred["race"], target["race"]
        ),
    }

    age_main_loss = age_strategy.loss(
        pred["age"], target["age"]
    )
    age_mae       = age_strategy.mae(
        pred["age"], target["age"]
    )

    losses["age"] = age_main_loss + (age_mae_weight * age_mae)

    loss_weights = loss_weights or DEFAULT_LOSS_WEIGHTS
    total_loss = sum(loss_weights[k] * losses[k] for k in losses)

    return total_loss, losses
