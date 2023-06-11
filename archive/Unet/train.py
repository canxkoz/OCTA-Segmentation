import os
import torch

import time
import copy
from skimage.metrics import structural_similarity as ssim
from collections import defaultdict

from tqdm import tqdm
from evaluation import print_metrics, calc_loss


def train_model(
    model,
    optimizer,
    scheduler,
    patience,
    log_file,
    dataloaders,
    device,
    dtype,
    output_save_dir,
    model_name,
    loss_type="mse",
    num_epochs=25,
):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e15
    file = open(log_file, "a")
    for epoch in range(1, num_epochs + 1):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print("-" * 10)
        file.write("Epoch {}/{}\n".format(epoch, num_epochs))
        file.write("-" * 10)
        since = time.time()
        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                for param_group in optimizer.param_groups:
                    print("\nLR", param_group["lr"])
                    file.write(f"\nLR {param_group['lr']}\n")
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0
            for inputs, labels, names in tqdm(dataloaders[phase]):
                inputs = inputs.to(device).type(dtype)
                labels = labels.to(device).type(dtype)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    amin = torch.amin(outputs, dim=(-2, -1))
                    outputs = torch.sub(
                        outputs, amin.reshape((amin.shape[0], amin.shape[1], 1, 1))
                    )

                    loss = calc_loss(outputs, labels, metrics, loss_type=loss_type)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print(print_metrics(metrics, epoch_samples, phase))
            file.write(print_metrics(metrics, epoch_samples, phase) + "\n")
            epoch_loss = metrics["loss"] / epoch_samples

            # deep copy the model
            if phase == "val" and epoch_loss <= best_loss:
                print("saving best model")
                file.write("saving best model\n")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

                save_dir = os.path.join(output_save_dir, "models/")
                os.makedirs(save_dir, exist_ok=True)
                torch.save(best_model_wts, os.path.join(save_dir, model_name))
            """if phase == 'val':
                valid_loss = epoch_loss
                scheduler.step(epoch_loss)"""
            if phase == "val":
                scheduler.step()
        time_elapsed = time.time() - since
        print("{:.0f}m {:.0f}s\n".format(time_elapsed // 60, time_elapsed % 60))
        file.write("{:.0f}m {:.0f}s\n\n".format(time_elapsed // 60, time_elapsed % 60))

    print("Best val loss: {:4f}".format(best_loss))
    file.write("Best val loss: {:4f}\n".format(best_loss))
    file.close()
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
