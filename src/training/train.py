import torch

from .losses import compute_losses
from .metrics import (init_tracker, update_tracker,
                    compute_overall_metrics, compute_subgroup_metric)

tasks = ["gender", "age", "race"]

def train_loop(dataloader, model, loss_funcs, loss_weights, optimizer, device):
    size = len(dataloader.dataset)

    model.train()

    running_loss = 0.0
    total_samples = 0
    running_task_losses = {task: 0.0 for task in tasks}

    tracker = init_tracker()
    print("\n----------TRAIN----------")
    for batch, (X,Y) in enumerate(dataloader):
      
        X = X.to(device)
        Y = { key: value.to(device) for key, value in Y.items() }
        
        batch_size = X.size(0)

        pred = model(X)

        total_loss, task_losses = compute_losses(pred, Y, loss_funcs, loss_weights)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        update_tracker(tracker, pred, Y)
        running_loss  += total_loss.item() * batch_size

        for task in tasks:
            running_task_losses[task] += task_losses[task].item() * batch_size

        total_samples += batch_size

        if batch % 100 == 0:
            current = batch * batch_size + batch_size
            print(f" loss: {total_loss.item():>7f}  [{current:>5d}/{size:>5d}]\n")

        # current = batch * batch_size + batch_size
        # print(f"loss: {total_loss.item():>7f}  [{current:>5d}/{size:>5d}]")
        # if batch == 2:
        #     break
    overall  = compute_overall_metrics(tracker)   
    avg_loss = running_loss / total_samples
    avg_task_losses = {k: v / total_samples for k, v in running_task_losses.items()}
    
    
    print(f" avg loss:        {avg_loss:.4f}")
    print(f" avg age loss:    {avg_task_losses['age']:.2f}")
    print(f" avg gender loss: {avg_task_losses['gender']:.2f}")
    print(f" avg race loss:   {avg_task_losses['race']:.2f}\n")

    print(f" gender: acc={overall['gender']['accuracy']:.3f}  f1={overall['gender']['f1']:.3f}")
    print(f" age:    acc={overall['age']['accuracy']:.3f}  f1={overall['age']['f1']:.3f}  mae={overall['age']['mae']:.3f}")
    print(f" race:   acc={overall['race']['accuracy']:.3f}  f1={overall['race']['f1']:.3f}")

    return avg_loss, avg_task_losses 



def test_loop(dataloader, model, loss_funcs, loss_weights, device):
    model.eval()

    running_loss  = 0.0
    running_task_losses = {task: 0.0 for task in tasks}
    total_samples = 0
    # correct_pred_num  = {"gender": 0,   "age": 0,   "race": 0}
    # running_task_losses = {"gender": 0.0, "age": 0.0, "race": 0.0}
    
    tracker = init_tracker()

    print("\n----------TEST----------")
    with torch.no_grad():
        for batch, (X, Y) in enumerate(dataloader):
            X = X.to(device)
            Y = {key: value.to(device) for key, value in Y.items()}
            batch_size = X.size(0)

            pred = model(X)
            total_loss, task_losses = compute_losses(pred, Y, loss_funcs, loss_weights)

            running_loss += total_loss.item() * batch_size

            for task in tasks:
                running_task_losses[task] += task_losses[task].item() * batch_size

            total_samples += batch_size

            update_tracker(tracker, pred, Y)

            # gender_preds = (torch.sigmoid(pred["gender"].squeeze(1)) > 0.5).long()
            # correct_pred_num["gender"] += (gender_preds == Y["gender"].long()).sum().item()

            # age_preds = pred["age"].argmax(dim=1)
            # correct_pred_num["age"] += (age_preds == Y["age"]).sum().item()

            # race_preds = pred["race"].argmax(dim=1)
            # correct_pred_num["race"] += (race_preds == Y["race"]).sum().item()

    avg_loss = running_loss / total_samples
    avg_task_losses = {k: v / total_samples for k, v in running_task_losses.items()}

    overall = compute_overall_metrics(tracker)
    subgroups_accuracy = compute_subgroup_metric(tracker)

    # avg_task_losses = {k: v / total_samples for k, v in running_task_losses.items()}
    # accuracy = {k: v / total_samples for k, v in correct_pred_num.items()}
    
    print(f" avg loss: {avg_loss:.4f}")
    # print(f"Epoch {t+1}: train_loss={train_loss:.4f}  test_loss={test_loss:.4f}")
    print(f" gender: acc={overall['gender']['accuracy']:.3f}  f1={overall['gender']['f1']:.3f}")
    print(f" age:    acc={overall['age']['accuracy']:.3f}  f1={overall['age']['f1']:.3f}   mae={overall['age']['mae']:.3f}")
    print(f" race:   acc={overall['race']['accuracy']:.3f}  f1={overall['race']['f1']:.3f}")
    # for task in ["gender", "age", "race"]:
    #     print(f"  {task}: loss={avg_task_losses[task]:.4f}  acc={accuracy[task]:.4f}")

    return avg_loss, avg_task_losses, overall, subgroups_accuracy