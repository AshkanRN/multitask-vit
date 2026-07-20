import torch
import torch.nn as nn
import numpy as np

from sklearn.utils.class_weight import compute_class_weight

DEFAULT_LOSS_WEIGHTS = {
    "gender": 1.0,
    "age": 1.0,
    "race": 1.0
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



def get_loss_function(age_weights=None,race_weights=None):
    return {
        'gender': nn.BCEWithLogitsLoss(),
        'age'   : nn.CrossEntropyLoss(weight=age_weights, label_smoothing=0.1),
        'race'  : nn.CrossEntropyLoss(weight=race_weights)   
    }


def ordinal_age_mae(age_logits, age_target, num_classes=9):
    """
    Soft ordinal penalty: expected age (from softmax probs) vs true age bin.
    """
    probs = torch.softmax(age_logits, dim=1)
    
    age_values = torch.arange(num_classes, device=age_logits.device).float()
    expected_age = (probs * age_values).sum(dim=1)
    return torch.abs(expected_age - age_target.float()).mean()




def compute_losses(pred, target, loss_funcs, age_mae_weight=AGE_MAE_WEIGHT, loss_weights=None):
    losses = {}

     # gender
    losses["gender"] = loss_funcs["gender"](
        pred["gender"].squeeze(1),
        target["gender"].float()
    )

    # age
    age_cross_entropy = loss_funcs["age"](
        pred["age"],
        target["age"]
    )   
    age_mae = ordinal_age_mae(pred["age"], target["age"], num_classes=pred["age"].shape[1])
    losses["age"] = age_cross_entropy + age_mae_weight * age_mae

    # race
    losses["race"] = loss_funcs["race"](
        pred["race"],
        target["race"]
    )


    if loss_weights is None:
        loss_weights = DEFAULT_LOSS_WEIGHTS


    total_loss = (
        loss_weights["gender"] * losses["gender"]
        +
        loss_weights["age"] * losses["age"]
        +
        loss_weights["race"] * losses["race"]
    )


    return total_loss, losses
