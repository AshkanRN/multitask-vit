# age strategy

import torch
import torch.nn as nn
import torch.nn.functional as F

# single source of truth for the age head's output size
AGE_OUTPUT_DIMS = {
    "ce": 9, 
    "corn": 8
}


# ---- low-level math (unchanged from your losses.py) ----

def ordinal_age_mae(age_logits, age_target, num_classes):
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
    probas = torch.sigmoid(age_logits)
    cumprobs = torch.cumprod(probas, dim=1)
    expected_age = torch.sum(cumprobs, dim=1)
    return torch.abs(expected_age - age_target.float()).mean()


# ---- strategies: everything else in the codebase only talks to these ----

class CEAgeStrategy:
    name = "ce"
    output_dim = AGE_OUTPUT_DIMS["ce"]

    def __init__(self, weights=None, label_smoothing=0.1):
        self._loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)

    def loss(self, logits, target):
        return self._loss_fn(logits, target)

    def mae(self, logits, target):
        return ordinal_age_mae(logits, target, num_classes=self.output_dim)

    def decode(self, logits):
        return logits.argmax(dim=1)

    def probs(self, logits):
        """Returns (per_class_probs, sortable). sortable=True means argmax == prediction."""
        return torch.softmax(logits, dim=1), True


class CORNAgeStrategy:
    name = "corn"
    output_dim = AGE_OUTPUT_DIMS["corn"]

    def __init__(self, num_classes=9):
        self.num_classes = num_classes

    def loss(self, logits, target):
        return corn_loss(logits, target, num_classes=self.num_classes)

    def mae(self, logits, target):
        return corn_expected_age_mae(logits, target)

    def decode(self, logits):
        return corn_label_from_logits(logits).long()

    def probs(self, logits):
        """Threshold pass-probabilities — monotonically decreasing, NOT argmax-able."""
        return torch.sigmoid(logits), False


def get_age_strategy(age_loss_type, age_weights=None, num_age_classes=9):
    if age_loss_type == "ce":
        return CEAgeStrategy(weights=age_weights)
    elif age_loss_type == "corn":
        return CORNAgeStrategy(num_classes=num_age_classes)
    raise ValueError(f"Unknown age_loss_type: {age_loss_type!r}")