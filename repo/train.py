import torch
from DataLoader import Data_Reg_Binary
from loss import calc_loss

from Model import UNet, UNet_mt
from collections import defaultdict
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import os 
import time

batch_size = 2
num_workers = 20

train_path = '/home/caki/desktop/projects/unet/unet-tf/rose2/train'
val_path = '/home/caki/desktop/projects/unet/unet-tf/rose2/test'
input_size = (512,512)
use_cuda = True
device = "cuda:0"
dtype = torch.cuda.FloatTensor
num_class = 1
ch = 1
output_save_dir = 'exp5'



def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    return ("{}: {}".format(phase, ", ".join(outputs)))

def plot_loss_functions(output_save_dir, train_loss_list, val_loss_list):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(np.arange(len(train_loss)),train_loss,label='train loss')
    plt.plot(np.arange(len(val_loss)),val_loss,label='val loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_save_dir,'loss.png'))
    plt.cla()  

def train_model(model, output_save_dir, dataloaders, optimizer, scheduler, patience, loss_type='mse', num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e15
    if not os.path.exists(output_save_dir):
        os.mkdir(output_save_dir)
    log_file = os.path.join(output_save_dir, "logs.txt")

    file = open(log_file, 'a')
    train_loss_list = []
    val_loss_list = []
    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        file.write('Epoch {}/{}'.format(epoch, num_epochs))
        file.write("\n")
        file.write('-' * 10)
        file.write("\n")
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                    file.write(f"LR {param_group['lr']}")
                    file.write("\n")
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0
            for inputs, label_mask, label_dist in tqdm(dataloaders[phase]):

                inputs = inputs.to(device).type(dtype)
                label_mask = label_mask.to(device).type(dtype)
                label_dist = label_dist.to(device).type(dtype)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    output_mask, output_dist = model(inputs)

                    amin = torch.amin(output_dist, dim=(-2,-1))
                    output_dist = torch.sub(output_dist, amin.reshape((amin.shape[0], amin.shape[1], 1, 1)))

                    loss1 = calc_loss(output_mask, label_mask, metrics,
                                     loss_type='bce')

                    loss2 = calc_loss(output_dist, label_dist, metrics,
                                     loss_type='mse')

                    loss = loss1 + loss2
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print(print_metrics(metrics, epoch_samples, phase))
            file.write(print_metrics(metrics, epoch_samples, phase))
            file.write("\n")
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss <= best_loss:
                print("saving best model")
                file.write("saving best model")
                file.write("\n")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                model_name = 'epoch{}.pt'.format(epoch)
                save_dir = os.path.join(output_save_dir, 'models/')
                os.makedirs(save_dir, exist_ok=True)
                torch.save(best_model_wts, os.path.join(save_dir, model_name))
            if phase == 'val':
                valid_loss = epoch_loss
                scheduler.step(epoch_loss)
                val_loss_list.append(valid_loss)
            else:
                train_loss_list.append(epoch_loss)

        
        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s\n'.format(
            time_elapsed // 60, time_elapsed % 60))
        file.write('{:.0f}m {:.0f}s\n'.format(
            time_elapsed // 60, time_elapsed % 60))
        file.write("\n")

    print('Best val loss: {:4f}'.format(best_loss))
    file.write('Best val loss: {:4f}'.format(best_loss))
    file.write("\n")
    file.close()
    # load best model weights
    model.load_state_dict(best_model_wts)
    plot_loss_functions(output_save_dir,val_loss_list,train_loss_list)

    return model


def main():

    # Training data loader
    train_dataset = Data_Reg_Binary(train_path, ch, scale_size=input_size)
    val_dataset = Data_Reg_Binary(val_path, ch, scale_size=input_size)
    print('Train set size:', len(train_dataset))
    print('Val set size:', len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)

    torch.autograd.set_detect_anomaly(True)

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }


    model = UNet_mt(ch, num_class, use_cuda)
    if use_cuda:
        print('Gpu available')
        print(torch.cuda.get_device_name(0))
        model.to(device=device)
    else:
        model.to(device="cpu")
    print(model)

    #optimizers
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    

    model = train_model(model, output_save_dir, dataloaders, optimizer, lr_scheduler, 
                    patience=30, num_epochs=50, loss_type='mse')

if __name__ == "__main__":
    main()
