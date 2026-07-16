import torch
import math

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error)


tasks = ['gender', 'age', 'race']

def init_tracker():
    return {
        'preds': {task: [] for task in tasks},
        'targets': {task: [] for task in tasks}
    }


def update_tracker(tracker, pred, target):
    gender_preds = (torch.sigmoid(pred["gender"].squeeze(1)) > 0.5).long()
    age_preds    = pred["age"].argmax(dim=1)
    race_preds   = pred["race"].argmax(dim=1)
 
    tracker["preds"]["gender"].extend(gender_preds.cpu().tolist())
    tracker["preds"]["age"].extend(age_preds.cpu().tolist())
    tracker["preds"]["race"].extend(race_preds.cpu().tolist())
 
    tracker["targets"]["gender"].extend(target["gender"].cpu().long().tolist())
    tracker["targets"]["age"].extend(target["age"].cpu().tolist())
    tracker["targets"]["race"].extend(target["race"].cpu().tolist())


def compute_overall_metrics(tracker):  
    results = {}

    # calc accuracy, precision, recall, F1 for all tasks

    for task in tasks:
        predictions = tracker['preds'][task]
        targets = tracker['targets'][task]

        if task == "gender":
            average = "binary"
        else:
            average = "macro"

        results[task] = {
            "accuracy":  accuracy_score(targets, predictions),
            "precision": precision_score(targets, predictions, average=average, zero_division=0),
            "recall":    recall_score(targets, predictions, average=average, zero_division=0),
            "f1":        f1_score(targets, predictions, average=average, zero_division=0),
        }


    # cals MSE, RMSE and MAE only for age

    age_predictions = tracker['preds']['age']
    age_targets     = tracker['targets']['age']

    mse = mean_squared_error(age_targets, age_predictions)

    results['age']['mse']  = mse
    results['age']['rmse'] = math.sqrt(mse)
    results["age"]["mae"]  = mean_absolute_error(age_targets, age_predictions)

    return results

def subgroup_accuracy(preds, labels, groups):

    results = {}
    for g in sorted(set(groups)):
        idx = [i for i, gv in enumerate(groups) if gv == g]
        p = [preds[i] for i in idx]
        t = [labels[i] for i in idx]
        results[g] = accuracy_score(t, p)

    return results

def compute_subgroup_metric(tracker):

    task_combo = [
        ("gender", "race"),
        ("gender", "age"),
        ("age", "gender"),
        ("age", "race"),
        ("race", "gender"),
        ("race", "age"),
    ]
    results = {}

    for target_task, group_task in task_combo:
        key = f"{target_task} by {group_task}"
        
        results[key] = subgroup_accuracy(
            tracker["preds"][target_task],
            tracker["targets"][target_task],
            tracker["targets"][group_task]
        )

    return results