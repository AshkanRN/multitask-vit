import torch

from losses import compute_losses


def train_loop(dataloader, model, loss_funcs, loss_weights, optimizer, device):
    size        = len(dataloader.dataset)

    model.train()

    running_loss = 0.0
    total_samples = 0

    for batch, (X,Y) in enumerate(dataloader):

        X = X.to(device)
        Y = { key: value.to(device) for key, value in Y.items() }
        
        batch_size = X.size(0)

        pred = model(X)

        total_loss, task_losses = compute_losses(pred,Y,loss_funcs,loss_weights)

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()


        running_loss  += total_loss.item() * batch_size
        total_samples += batch_size

        if batch % 100 == 0:
            current = batch * batch_size + batch_size
            print(f"loss: {total_loss.item():>7f}  [{current:>5d}/{size:>5d}]")

    avg_loss = running_loss / total_samples
    print(f"Train avg loss: {avg_loss:.4f}")
    return avg_loss 



def test_loop(dataloader, model, loss_funcs, loss_weights, device):
    model.eval()

    running_loss  = 0.0
    running_task_losses = {"gender": 0.0, "age": 0.0, "race": 0.0}
    correct_pred_num    = {"gender": 0,   "age": 0,   "race": 0}
    total_samples = 0

    with torch.no_grad():
        for X, Y in dataloader:
            X = X.to(device)
            Y = {key: value.to(device) for key, value in Y.items()}
            batch_size = X.size(0)

            pred = model(X)
            total_loss, task_losses = compute_losses(pred, Y, loss_funcs, loss_weights)

            running_loss += total_loss.item() * batch_size
            total_samples += batch_size

            for task_name, loss_tensor in task_losses.items():
                running_task_losses[task_name] += loss_tensor.item() * batch_size

            gender_preds = (torch.sigmoid(pred["gender"].squeeze(1)) > 0.5).long()
            correct_pred_num["gender"] += (gender_preds == Y["gender"].long()).sum().item()

            age_preds = pred["age"].argmax(dim=1)
            correct_pred_num["age"] += (age_preds == Y["age"]).sum().item()

            race_preds = pred["race"].argmax(dim=1)
            correct_pred_num["race"] += (race_preds == Y["race"]).sum().item()

    avg_loss = running_loss / total_samples
    avg_task_losses = {k: v / total_samples for k, v in running_task_losses.items()}
    accuracy = {k: v / total_samples for k, v in correct_pred_num.items()}

    print(f"Test avg loss: {avg_loss:.4f}")
    for task in ["gender", "age", "race"]:
        print(f"  {task}: loss={avg_task_losses[task]:.4f}  acc={accuracy[task]:.4f}")

    return avg_loss, avg_task_losses, accuracy