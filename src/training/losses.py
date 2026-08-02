import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from functools import partial
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



def get_loss_function(age_loss_type="ce", age_weights=None, race_weights=None, num_age_classes=9):
    if age_loss_type == "corn":
        age_loss_fn = partial(corn_loss, num_classes=num_age_classes)
    elif age_loss_type == "ce":
        age_loss_fn = nn.CrossEntropyLoss(weight=age_weights, label_smoothing=0.1)
    else:
        raise ValueError(f"Unknown age_loss_type: {age_loss_type}")

    return {
        'gender': nn.BCEWithLogitsLoss(),
        'age'   : age_loss_fn,
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


def corn_loss(logits, y_train, num_classes):
    """CORN loss (Shi et al., 2021). logits: (N, num_classes-1), y_train: (N,) int labels."""
    sets = []
    for i in range(num_classes - 1):
        label_mask = y_train > i - 1
        label_tensor = (y_train[label_mask] > i).to(torch.int64)
        sets.append((label_mask, label_tensor))

    num_examples = 0
    losses = 0.
    for task_index, (train_examples, train_labels) in enumerate(sets):
        if len(train_labels) < 1:
            continue
        num_examples += len(train_labels)
        pred = logits[train_examples, task_index]
        loss = -torch.sum(
            F.logsigmoid(pred) * train_labels
            + (F.logsigmoid(pred) - pred) * (1 - train_labels)
        )
        losses += loss

    return losses / num_examples


def corn_label_from_logits(logits):
    """Hard label decode: number of thresholds passed."""
    probas = torch.sigmoid(logits)
    cumprobs = torch.cumprod(probas, dim=1)
    predict_levels = cumprobs > 0.5
    return torch.sum(predict_levels, dim=1)


def corn_expected_age_mae(age_logits, age_target):
    """Soft/differentiable expected-bin MAE, CORN version of your old ordinal_age_mae."""
    probas = torch.sigmoid(age_logits)
    cumprobs = torch.cumprod(probas, dim=1)
    expected_age = torch.sum(cumprobs, dim=1)
    return torch.abs(expected_age - age_target.float()).mean()



def compute_losses(pred, target, loss_funcs, age_loss_type="ce", age_mae_weight=AGE_MAE_WEIGHT, loss_weights=None):
    losses = {}

     # gender
    losses["gender"] = loss_funcs["gender"](
        pred["gender"].squeeze(1),
        target["gender"].float()
    )

    # age
    if age_loss_type == "corn":
        age_main_loss = loss_funcs["age"](pred["age"], target["age"])
        age_mae = corn_expected_age_mae(pred["age"], target["age"])
    else:
        age_main_loss = loss_funcs["age"](pred["age"], target["age"])
        age_mae = ordinal_age_mae(pred["age"], target["age"], num_classes=pred["age"].shape[1])

    losses["age"] = age_main_loss + age_mae_weight * age_mae
    # age_corn = loss_funcs["age"](pred["age"], target["age"])
    # age_mae  = corn_expected_age_mae(pred["age"], target["age"])
    # losses["age"] = age_corn + age_mae_weight * age_mae

    # age_cross_entropy = loss_funcs["age"](
    #     pred["age"],
    #     target["age"]
    # )   
    # age_mae = ordinal_age_mae(pred["age"], target["age"], num_classes=pred["age"].shape[1])
    # losses["age"] = age_cross_entropy + age_mae_weight * age_mae

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
