import torch.nn as nn

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
