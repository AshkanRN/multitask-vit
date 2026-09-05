import torch
from tqdm.auto import tqdm 
from torch.amp import autocast

from .losses import compute_losses
from .metrics import (init_tracker, update_tracker,
                    compute_overall_metrics, compute_subgroup_metric)

tasks = ["gender", "age", "race"]

def train_loop(dataloader, model, loss_funcs, loss_weights,
                optimizer, device, epoch, epochs, scaler=None):
    age_strategy = loss_funcs["age_strategy"]

    model.train()

    running_loss = 0.0
    total_samples = 0
    running_task_losses = {task: 0.0 for task in tasks}

    tracker = init_tracker()
    progress_bar = tqdm(dataloader, desc=f"Train Epoch {epoch}/{epochs}", leave=False)

    for batch, (X, Y) in enumerate(progress_bar):
      
        X = X.to(device)
        Y = { key: value.to(device) for key, value in Y.items() }
        
        batch_size = X.size(0)

    
        optimizer.zero_grad()

        with autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            pred = model(X)
            total_loss, task_losses = compute_losses(
                pred, Y, loss_funcs, loss_weights=loss_weights
            )

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()


        # without scaler
        # pred = model(X)

        # total_loss, task_losses = compute_losses(
        #     pred, Y, loss_funcs, loss_weights=loss_weights
        # )

        # total_loss.backward()
        # optimizer.step()




        update_tracker(tracker, pred, Y, age_strategy)
        running_loss  += total_loss.item() * batch_size

        for task in tasks:
            running_task_losses[task] += task_losses[task].item() * batch_size

        total_samples += batch_size

        progress_bar.set_postfix({
            "loss": f"{total_loss.item():.4f}",
            "age": f"{task_losses['age'].item():.3f}",
            "gender": f"{task_losses['gender'].item():.3f}",
            "race": f"{task_losses['race'].item():.3f}",
        })

    overall  = compute_overall_metrics(tracker)   
    avg_loss = running_loss / total_samples
    avg_task_losses = {k: v / total_samples for k, v in running_task_losses.items()}
    
    print("-TRAIN")
    print(f" avg loss:        {avg_loss:.4f}")
    print(f" avg age loss:    {avg_task_losses['age']:.2f}")
    print(f" avg gender loss: {avg_task_losses['gender']:.2f}")
    print(f" avg race loss:   {avg_task_losses['race']:.2f}\n")

    print(f" gender: acc={overall['gender']['accuracy']:.3f}  f1={overall['gender']['f1']:.3f}")
    print(f" age:    acc={overall['age']['accuracy']:.3f}  f1={overall['age']['f1']:.3f}  mae={overall['age']['mae']:.3f}")
    print(f" race:   acc={overall['race']['accuracy']:.3f}  f1={overall['race']['f1']:.3f}")

    return avg_loss, avg_task_losses, overall 



def evaluate_loop(dataloader, model, loss_funcs, loss_weights, device, epoch, epochs,use_amp=True):

    age_strategy = loss_funcs["age_strategy"]

    model.eval()

    running_loss  = 0.0
    running_task_losses = {task: 0.0 for task in tasks}
    total_samples = 0

    
    tracker = init_tracker()
    progress_bar = tqdm(dataloader, desc=f"validation Epoch {epoch}/{epochs}", leave=False)

    with torch.no_grad():
        for batch, (X,Y) in enumerate(progress_bar):
            X = X.to(device)
            Y = {key: value.to(device) for key, value in Y.items()}
            batch_size = X.size(0)

            
            with autocast("cuda", dtype=torch.float16, enabled=(torch.cuda.is_available() and use_amp)):
                pred = model(X)

                total_loss, task_losses = compute_losses(
                    pred, Y, loss_funcs, loss_weights=loss_weights
                )


            # without scaler
            # pred = model(X)

            # total_loss, task_losses = compute_losses(
            #     pred, Y, loss_funcs, loss_weights=loss_weights
            # )

            running_loss += total_loss.item() * batch_size

            for task in tasks:
                running_task_losses[task] += task_losses[task].item() * batch_size

            total_samples += batch_size

            update_tracker(tracker, pred, Y, age_strategy)

            progress_bar.set_postfix({
                "loss": f"{total_loss.item():.4f}"
            })


    avg_loss = running_loss / total_samples
    avg_task_losses = {k: v / total_samples for k, v in running_task_losses.items()}
    overall = compute_overall_metrics(tracker)
    subgroups_accuracy = compute_subgroup_metric(tracker)
    
    print("-VALIDATION")
    print(f" avg loss: {avg_loss:.4f}")
    
    print(f" gender: acc={overall['gender']['accuracy']:.3f}  f1={overall['gender']['f1']:.3f}")
    print(f" age:    acc={overall['age']['accuracy']:.3f}  f1={overall['age']['f1']:.3f}   mae={overall['age']['mae']:.3f}")
    print(f" race:   acc={overall['race']['accuracy']:.3f}  f1={overall['race']['f1']:.3f}")


    return avg_loss, avg_task_losses, overall, subgroups_accuracy