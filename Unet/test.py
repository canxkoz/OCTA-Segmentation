import os
import torch
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from evaluation import print_metrics, calc_loss


def test_model(
    model, log_file, dataloaders, device, dtype, output_save_dir, loss_type="mse"
):
    model.eval()  # Set model to evaluate mode
    file = open(log_file, "a")
    metrics = defaultdict(float)
    epoch_samples = 0
    for inputs, labels, names in tqdm(dataloaders["val"]):
        inputs = inputs.to(device).type(dtype)
        labels = labels.to(device).type(dtype)

        outputs = model(inputs)
        amin = torch.amin(outputs, dim=(-2, -1))
        outputs = torch.sub(outputs, amin.reshape((amin.shape[0], amin.shape[1], 1, 1)))
        # outputs = outputs.reshape(inputs.shape)
        for i, out in enumerate(outputs):
            np_out = out.detach().cpu().numpy() * 255
            img = Image.fromarray(np_out[0]).convert("L")

            img_name = names[i]
            imgs_dir = os.path.join(output_save_dir, "predictions")
            os.makedirs(imgs_dir, exist_ok=True)

            # img.save(os.path.join(imgs_dir, img_name))

            fig = plt.figure(figsize=(10, 7))
            rows, columns = 1, 3

            fig.add_subplot(rows, columns, 1)
            # originalPath = os.path.join(dataPath, "test/original", img_name)
            input_np = np.asarray(Image.open(img_name))
            plt.imshow(input_np, cmap="gray")
            plt.axis("off")
            plt.title("Original")

            fig.add_subplot(rows, columns, 2)
            gtPath = img_name.replace("original", "gt")
            gt_np = np.asarray(Image.open(gtPath))
            plt.imshow(gt_np, cmap="gray")
            plt.axis("off")
            plt.title("Ground truth")

            fig.add_subplot(rows, columns, 3)
            plt.imshow(np_out[0], cmap="gray")
            plt.axis("off")
            plt.title("Prediction")

            plt.savefig(os.path.join(imgs_dir, "plot_" + img_name.split("\\")[-1]))
            plt.close(fig)
        loss = calc_loss(outputs, labels, metrics, loss_type=loss_type)
        # statistics
        epoch_samples += inputs.size(0)

    print(print_metrics(metrics, epoch_samples, "val"))
    file.write(print_metrics(metrics, epoch_samples, "val") + "\n")
    file.close()
