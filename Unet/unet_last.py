import os
import torch
import model as pytorch_unet
import vanilla_unet as vanilla_unet
from torch.utils.data import DataLoader
from test import test_model
from train import train_model
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
from dataset import build_dataset


dataPath = "data\\ROSE-2"
savePathPrefix = "./"

trInputPath = dataPath + "\\train\\"
valInputPath = dataPath + "\\test\\"

outputPathtr = dataPath + "\\golds\\"
outputPathval = dataPath + "\\golds\\"

imagePostfix = ".png"
goldPostfix = ".tif"

batch_size = 1
num_workers = 20

train_dataset = build_dataset(dataPath, channel=1, isTraining=True)
val_dataset = build_dataset(dataPath, channel=1, isTraining=False)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)

torch.autograd.set_detect_anomaly(True)

dataloaders = {"train": train_loader, "val": val_loader}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
reshape_size = 512

dtype = torch.cuda.FloatTensor

parser = argparse.ArgumentParser()
parser.add_argument(
    "--loss-type", help="The loss type (mse / rmse / l1loss)", default="mse"
)
parser.add_argument(
    "--weight-decay",
    help="Float value of weight decay for optimizer",
    default=0.000000001,
    type=float,
)
parser.add_argument(
    "--output-dir",
    help="Directory that outputs, logs and models will be saved.",
    default="outputs",
)
args = parser.parse_args()

run = args.loss_type
fold = 2
model_name = "UNet_fold{}_run_{}.pt".format(fold, run)

output_save_dir = os.path.join(args.output_dir, args.loss_type)
os.makedirs(output_save_dir, exist_ok=True)
log_file = os.path.join(output_save_dir, "logs.txt")
num_class = 1

# model = vanilla_unet.UNet(num_class)
####################
model_name = "can/vanilla_unet_epoch150.pt"

device = torch.device("cuda")
dtype = torch.cuda.FloatTensor
model = vanilla_unet.UNet(1).to(device)
model.load_state_dict(torch.load(model_name))
#####################
model_name = "vanilla_unet_epoch250.pt"

model = model.to(device)

optimizer_ft = optim.Adam(model.parameters(), lr=1e-4)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.9)

model = train_model(
    model,
    optimizer_ft,
    exp_lr_scheduler,
    patience=30,
    log_file=log_file,
    dataloaders=dataloaders,
    device=device,
    dtype=dtype,
    output_save_dir=output_save_dir,
    model_name=model_name,
    loss_type=args.loss_type,
    num_epochs=50,
)


test_model(
    model, log_file, dataloaders, device, dtype, output_save_dir, loss_type="mse"
)
