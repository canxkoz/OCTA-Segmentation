import torch
from DataLoader import Data_Reg_Binary, Data_Binary, Data_Reg_Fourier1
from loss import calc_loss, MultitaskUncertaintyLoss
from torchvision.utils import make_grid, save_image

from Model import UNet, UNet_multitask, UNet_attention, UNet_fourier1
from collections import defaultdict
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import os
import time
import numpy as np
import argparse
import yaml
import torch.nn.functional as F
import matplotlib.pyplot as plt
from Trainer import Trainer

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('config', help='the config path')
    args = ap.parse_args()
    return args


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    return ("{}: {}".format(phase, ", ".join(outputs)))


def plot_loss_functions(output_save_dir, train_loss, val_loss, name):
    plt.figure(figsize=(8, 4))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(np.arange(len(train_loss)), train_loss, label='train loss')
    plt.plot(np.arange(len(val_loss)), val_loss, label='val loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_save_dir, '{}.png'.format(name)))
    plt.cla()


def check_input(dataloaders, titles=["Input", 'Target']):
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    
    if len(train_batch) == 3:
        img, target, dist = train_batch
        number_of_batch = img.shape[0]
        print('image shape:', img.shape)
        print('binary label shape:', target.shape)
        print('dist label shape:', dist.shape)
        grid_img1 = make_grid(img)
        grid_img2 = make_grid(target)
        grid_img3 = make_grid(dist)
        ulti = make_grid([grid_img1, grid_img2, grid_img3], nrow=1)
        save_image(ulti,'train_batch.png')

        img, target, dist = val_batch
        number_of_batch = img.shape[0]
        print('image shape:', img.shape)
        print('binary label shape:', target.shape)
        print('dist label shape:', dist.shape)
        grid_img1 = make_grid(img)
        grid_img2 = make_grid(target)
        grid_img3 = make_grid(dist)
        ulti = make_grid([grid_img1, grid_img2, grid_img3], nrow=1)
        save_image(ulti, 'val_batch.png')

    else:
        img, target = train_batch
        number_of_batch = img.shape[0]
        print('image shape:', img.shape)
        print('binary label shape:', target.shape)
        grid_img1 = make_grid(img)
        grid_img2 = make_grid(target)
        ulti = make_grid([grid_img1, grid_img2], nrow=1)
        save_image(ulti, 'train_batch.png')

        img, target = val_batch
        number_of_batch = img.shape[0]
        print('image shape:', img.shape)
        print('binary label shape:', target.shape)
        grid_img1 = make_grid(img)
        grid_img2 = make_grid(target)
        ulti = make_grid([grid_img1, grid_img2], nrow=1)
        save_image(ulti, 'val_batch.png')


def train_model_multi_task(model, dtype, device, output_save_dir, dataloaders, optimizer, lr_scheduler, patience, loss_function, accuracy_metric, start_epoch, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e15
    if not os.path.exists(output_save_dir):
        os.mkdir(output_save_dir)
    log_file = os.path.join(output_save_dir, "logs.txt")

    file = open(log_file, 'a')
    train_loss_list = []
    val_loss_list = []

    train_loss_list_1 = []
    val_loss_list_1 = []

    train_loss_list_2 = []
    val_loss_list_2 = []

    log_var_task1 = torch.zeros((1,), requires_grad=True)
    log_var_task2 = torch.zeros((1,), requires_grad=True)
    params = ([p for p in model.parameters()] +
              [log_var_task1] + [log_var_task2])
    loss_combiner = MultitaskUncertaintyLoss()

    optimizer = optim.Adam(params, lr=1e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=30, min_lr=5e-6)
    
    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        file.write('Epoch {}/{}'.format(epoch, num_epochs))
        file.write("\n")
        file.write('-' * 10)
        file.write("\n")
        epoch_loss = 0.0
        val_score = 0.0
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

            batch_step = 0
            loss1_current_epoch = 0
            loss2_current_epoch = 0
            for inputs, label_mask, label_dist in tqdm(dataloaders[phase]):
                print('zzz')
                batch_step += 1
                inputs = inputs.to(device).type(dtype)
                label_mask = label_mask.to(device).type(dtype)
                label_dist = label_dist.to(device).type(dtype)

                # zero the parameter gradients
                optimizer.zero_grad()


                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    output_mask, output_dist = model(inputs)

                    # binary classification
                    loss1 = calc_loss(output_mask, label_mask,
                                      loss_type=loss_function)
                    
                    #regression
                    amin = torch.amin(output_dist, dim=(-2, -1))
                    output_dist = torch.sub(output_dist, amin.reshape(
                        (amin.shape[0], amin.shape[1], 1, 1)))
                    loss2 = calc_loss(output_dist, label_dist,
                                      loss_type='mse')

                    loss = loss_combiner(
                        [loss1, loss2], [log_var_task1, log_var_task2])

                    loss = loss.to(device)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                    else:
                        epoch_loss += loss.item()
                        val_score += calc_loss(output_mask, label_mask,
                                               loss_type=accuracy_metric)
            epoch_loss /= batch_step



            if phase == 'val':
                num_val_batches = len(dataloaders[phase])
                val_score = val_score / max(num_val_batches, 1)
                if lr_scheduler:
                    # lr_scheduler.step(epoch_loss)
                    lr_scheduler.step(val_score)

                val_loss_list.append(epoch_loss)
                val_loss_list_1.append(loss1_current_epoch/batch_step)
                val_loss_list_2.append(loss2_current_epoch/batch_step)
                print("Val loss on epoch %i: %f" % (epoch, epoch_loss))
                print("Val score on epoch %i: %f" % (epoch, val_score))
                file.write((f"Val loss on epoch {epoch}: {epoch_loss}"))
                file.write("\n")
                if epoch_loss <= best_loss:
                    print("saving best model")
                    file.write("saving best model")
                    file.write("\n")
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    model_name = 'epoch{}.pt'.format(epoch)
                    save_dir = os.path.join(output_save_dir, 'models/')
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(best_model_wts, os.path.join(
                        save_dir, model_name))
            else:
                std_1 = torch.exp(log_var_task1)**0.5
                std_2 = torch.exp(log_var_task2)**0.5
                print([std_1.item(), std_2.item()])
                train_loss_list.append(epoch_loss)
                train_loss_list_1.append(loss1_current_epoch/batch_step)
                train_loss_list_2.append(loss2_current_epoch/batch_step)
                print("Train loss on epoch %i: %f" % (epoch, epoch_loss))
                file.write((f"Train loss on epoch {epoch}: {epoch_loss}"))
                file.write("\n")

        torch.save(model.state_dict(), os.path.join(
            save_dir, 'last_epoch.pt'))

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
    plot_loss_functions(output_save_dir, train_loss_list,
                        val_loss_list, 'total')
    plot_loss_functions(output_save_dir, train_loss_list_1,
                        val_loss_list_1, 'bce')
    plot_loss_functions(output_save_dir, train_loss_list_2,
                        val_loss_list_2, 'mse')

    return model




def main(cfg):

    # model configs
    input_size = (cfg['model_config']['input_size'][1],
                  cfg['model_config']['input_size'][0])
    num_class = cfg['model_config']['num_class']
    ch = cfg['model_config']['channel']
    initial_filter_size = cfg['model_config']['initial_filter_size'][0]
    kernel_size = cfg['model_config']['kernel'][0]
    model_type = cfg['model_config']['model_type']

    # train configs
    batch_size = cfg['train_config']['batch_size'][0]
    num_workers = cfg['train_config']['num_workers']
    lr_rate = cfg['train_config']['lr_rate'][0]
    Epoch = cfg['train_config']['epochs']
    use_cuda = cfg['train_config']['use_cuda']
    loss_function = cfg['train_config']['loss']
    accuracy_metric = cfg['train_config']['accuracy']
    weight_decay = cfg['train_config']['weight_decay'][0]

    # dataset configs
    train_path = cfg['dataset_config']['train_path']
    val_path = cfg['dataset_config']['val_path']
    aug_rate = cfg['dataset_config']['aug_rate']
    output_save_dir = cfg['dataset_config']['save_dir']

    
    if model_type == 'single':
        train_dataset = Data_Binary(
                    train_path, ch, input_size=input_size)
        val_dataset = Data_Binary(val_path, ch, input_size=input_size)

        model = UNet(ch, num_class, initial_filter_size, use_cuda)
        
    elif model_type == 'multi_task':
        train_dataset = Data_Reg_Binary(
                    train_path, ch, input_size=input_size)
        val_dataset = Data_Reg_Binary(val_path, ch, input_size=input_size)
        model = UNet_multitask(ch, num_class, initial_filter_size, use_cuda)

    elif model_type == 'attention':
        train_dataset = Data_Binary(
            train_path, ch, input_size=input_size)
        val_dataset = Data_Binary(val_path, ch, input_size=input_size)
        model = UNet_attention(ch, num_class, initial_filter_size, use_cuda)

    elif model_type == 'fourier1':
        train_dataset = Data_Reg_Fourier1(
            train_path, ch, input_size=input_size)
        val_dataset = Data_Reg_Fourier1(val_path, ch, input_size=input_size)
        model = UNet_fourier1(ch, num_class, initial_filter_size, use_cuda)
    else:
        raise ValueError('Invalid model_type "%s"' % model_type)
    
    start_epoch = 1
    if cfg['resume']['flag']:
        model.load_state_dict(torch.load(cfg['resume']['path']))
        start_epoch = cfg['resume']['epoch']
    if use_cuda:
        print('Gpu available')
        print(torch.cuda.get_device_name(0))
        device = "cuda:0"
        dtype = torch.cuda.FloatTensor
        model.to(device=device)
    else:
        model.to(device="cpu")

    print('Train set size:', len(train_dataset))
    print('Val set size:', len(val_dataset))
    print(model)
    train_loader = DataLoader(
        train_dataset, batch_size,
        shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    check_input(dataloaders)
    # optimizers
    optimizer = optim.Adam(
        model.parameters(), lr=lr_rate, weight_decay=weight_decay)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=30)

    trainer = Trainer(model, model_type, dtype, device, output_save_dir, dataloaders, batch_size, optimizer,
                      patience=30, num_epochs=Epoch, loss_function=loss_function, accuracy_metric=accuracy_metric, lr_scheduler=lr_scheduler, start_epoch=start_epoch)
    best_model = trainer.train()
    

if __name__ == "__main__":
    args = parse_args()
    config_path = args.config
    # config_path = 'config.yml'
    with open(config_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    main(cfg)
