import copy
import os
import time
from tqdm import tqdm
import torch
from loss import calc_loss, MultitaskUncertaintyLoss
import numpy as np
import matplotlib.pyplot as plt
from torch import optim


class Trainer():
    def __init__(self, model, model_type, dtype, device, output_save_dir, dataloaders, batch_size, optimizer, patience, num_epochs, loss_function, accuracy_metric,  lr_scheduler=None, start_epoch=1):
        self.model = model
        self.dataloader = dataloaders
        self.optimizer = optimizer
        self.start_epoch = start_epoch
        self.num_epochs = num_epochs
        self.patience = patience
        self.lr_scheduler = lr_scheduler
        self.best_loss = 1e15
        self.phases = ["train", "val"]
        self.best_model = []
        self.best_val_score = 0
        self.batch_size = batch_size
        self.output_save_dir = output_save_dir
        self.dtype = dtype
        self.device = device
        self.loss_function = loss_function
        self.accuracy_metric = accuracy_metric
        self.train_loss_list = []
        self.val_loss_list = []
        self.model_type = model_type

        self.train_loss_list_1 = []
        self.val_loss_list_1 = []

        self.train_loss_list_2 = []
        self.val_loss_list_2 = []


    def plot_loss_functions(self, name):
        plt.figure(figsize=(8, 4))
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(np.arange(len(self.train_loss_list)),
                 self.train_loss_list, label='train loss')
        plt.plot(np.arange(len(self.val_loss_list)),
                 self.val_loss_list, label='val loss')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.output_save_dir, '{}.png'.format(name)))
        plt.cla()

        if self.train_loss_list_1:
            plt.figure(figsize=(8, 4))
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.plot(np.arange(len(self.train_loss_list_1)),
                     self.train_loss_list_1, label='train loss')
            plt.plot(np.arange(len(self.val_loss_list_1)),
                     self.val_loss_list_1, label='val loss')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(self.output_save_dir, '{}.png'.format('bce')))
            plt.cla()

        if self.train_loss_list_2:
            plt.figure(figsize=(8, 4))
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.plot(np.arange(len(self.train_loss_list_2)),
                     self.train_loss_list_2, label='train loss')
            plt.plot(np.arange(len(self.val_loss_list_2)),
                     self.val_loss_list_2, label='val loss')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(
                self.output_save_dir, '{}.png'.format('mse')))
            plt.cla()

    def train(self):
        if self.model_type == 'single':
            self.singe_train()
        
        elif self.model_type == 'multi_task':
            print('multi')

        elif self.model_type == 'attention':
            print('attention')

        elif self.model_type == 'fourier1':
            self.fourier1_train()
        else:
            raise ValueError('Invalid model_type "%s"' % self.model_type)

    def singe_train(self):
        if not os.path.exists(self.output_save_dir):
            os.mkdir(self.output_save_dir)
        log_file = os.path.join(self.output_save_dir, "logs.txt")

        file = open(log_file, 'a')

        for epoch in range(self.start_epoch, self.num_epochs+1):

            file.write('Epoch {}/{}'.format(epoch, self.num_epochs))
            file.write("\n")
            file.write('-' * 10)
            file.write("\n")

            since = time.time()

            # Each epoch has a training and validation phase
            for phase in self.phases:
                epoch_loss = 0.0
                val_score = 0.0
                if phase == 'train':
                    for param_group in self.optimizer.param_groups:
                        print("LR", param_group['lr'])
                        file.write(f"LR {param_group['lr']}")
                        file.write("\n")
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                batch_step = 0
                with tqdm(self.dataloader[phase], unit="batch") as tbar:
                    for inputs, label_mask in tbar:
                        tbar.set_description(f"Epoch {epoch}")
                        batch_step += 1
                        inputs = inputs.to(self.device).type(self.dtype)
                        label_mask = label_mask.to(
                            self.device).type(self.dtype)

                        # zero the parameter gradients
                        self.optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):

                            output_mask = self.model(inputs)

                            loss = calc_loss(output_mask, label_mask,
                                             loss_type=self.loss_function)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                self.optimizer.step()
                                epoch_loss += loss.item()
                                tbar.set_postfix(loss=epoch_loss/batch_step)

                            else:
                                epoch_loss += loss.item()
                                val_score += calc_loss(output_mask, label_mask,
                                                    loss_type=self.accuracy_metric)
                                tbar.set_postfix(loss=epoch_loss/batch_step,
                                                 accuracy=(val_score/(batch_step)))

                epoch_loss /= batch_step
                # deep copy the model
                if phase == 'val':
                    val_score /= batch_step
                    if self.lr_scheduler:
                        # lr_scheduler.step(epoch_loss)
                        self.lr_scheduler.step(val_score)

                    self.val_loss_list.append(epoch_loss)
                    print("Val loss on epoch %i: %f" % (epoch, epoch_loss))
                    print("Val score on epoch %i: %f" % (epoch, val_score))

                    file.write((f"Val loss on epoch {epoch}: {epoch_loss}"))
                    file.write((f"Val score on epoch {epoch}: {val_score}"))

                    file.write("\n")
                    if val_score >= self.best_val_score:
                        self.best_val_score = val_score
                        print("saving best model")
                        file.write("saving best model")
                        file.write("\n")
                        self.best_loss = epoch_loss
                        self.best_model = copy.deepcopy(
                            self.model.state_dict())
                        model_name = 'epoch{}.pt'.format(epoch)
                        save_dir = os.path.join(
                            self.output_save_dir, 'models/')
                        os.makedirs(save_dir, exist_ok=True)
                        torch.save(self.best_model, os.path.join(
                            save_dir, model_name))
                else:
                    self.train_loss_list.append(epoch_loss)
                    print("Train loss on epoch %i: %f" % (epoch, epoch_loss))
                    file.write((f"Train loss on epoch {epoch}: {epoch_loss}"))
                    file.write("\n")

            torch.save(self.model.state_dict(), os.path.join(
                save_dir, 'last_epoch.pt'))

            time_elapsed = time.time() - since
            print('{:.0f}m {:.0f}s\n'.format(
                time_elapsed // 60, time_elapsed % 60))
            file.write('{:.0f}m {:.0f}s\n'.format(
                time_elapsed // 60, time_elapsed % 60))
            file.write("\n")

        print('Best val loss: {:4f}'.format(self.best_loss))
        print('Best val score: {:4f}'.format(self.best_val_score))

        file.write('Best val loss: {:4f}'.format(self.best_loss))
        file.write('Best val score: {:4f}'.format(self.best_val_score))

        file.write("\n")
        file.close()
        # load best model weights
        self.model.load_state_dict(self.best_model)
        self.plot_loss_functions('total')

        return self.model

    def fourier1_train(self):
        self.best_model = copy.deepcopy(self.model.state_dict())
        if not os.path.exists(self.output_save_dir):
            os.mkdir(self.output_save_dir)
        log_file = os.path.join(self.output_save_dir, "logs.txt")

        file = open(log_file, 'a')

        total_memory = f'{torch.cuda.get_device_properties(0).total_memory/ 1E9 if torch.cuda.is_available() else 0:.3g}G'
        
        log_var_task1 = torch.zeros((1,), requires_grad=True)
        log_var_task2 = torch.zeros((1,), requires_grad=True)
        params = ([p for p in self.model.parameters()] +
                  [log_var_task1] + [log_var_task2])
        loss_combiner = MultitaskUncertaintyLoss()

        self.optimizer = optim.Adam(params, lr=1e-4)
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=30, min_lr=5e-6)

        for epoch in range(self.start_epoch, self.num_epochs+1):

            file.write('Epoch {}/{}'.format(epoch, self.num_epochs))
            file.write("\n")
            file.write('-' * 10)
            file.write("\n")

            since = time.time()

            # Each epoch has a training and validation phase
            for phase in self.phases:
                epoch_loss = 0.0
                loss1_current_epoch = 0
                loss2_current_epoch = 0
                val_score = 0.0
                if phase == 'train':
                    for param_group in self.optimizer.param_groups:
                        print("LR", param_group['lr'])
                        file.write(f"LR {param_group['lr']}")
                        file.write("\n")
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                batch_step = 0
                with tqdm(self.dataloader[phase], unit="batch") as tbar:
                    for inputs, label_mask, label_fdmap in tbar:
                        tbar.set_description(f"Epoch {epoch}")
                        batch_step += 1
                        inputs = inputs.to(self.device).type(self.dtype)
                        label_mask = label_mask.to(self.device).type(self.dtype)
                        label_fdmap = label_fdmap.to(self.device).type(self.dtype)

                        # zero the parameter gradients
                        self.optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):

                            output_mask, output_fdmap = self.model(inputs)

                            loss1 = calc_loss(output_mask, label_mask,
                                            loss_type=self.loss_function)
                            
                            loss2 = calc_loss(output_fdmap, label_fdmap,
                                              loss_type='mse')

                            loss = loss_combiner(
                                [loss1, loss2], [log_var_task1, log_var_task2])

                            loss = loss.to(self.device)

                            reserved = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                            mem = reserved + '/' + total_memory
                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                self.optimizer.step()
                                epoch_loss += loss.item()
                                tbar.set_postfix(
                                    loss=epoch_loss/batch_step,  memory=mem)
                                loss1_current_epoch += loss1.detach().item()
                                loss2_current_epoch += loss2.detach().item()
                            else:
                                epoch_loss += loss.item()
                                val_score += calc_loss(output_mask, label_mask,
                                                    loss_type=self.accuracy_metric)
                                tbar.set_postfix(loss=epoch_loss/batch_step,
                                                 accuracy=(val_score.item()/(batch_step)), memory=mem)
                                loss1_current_epoch += loss1.detach().item()
                                loss2_current_epoch += loss2.detach().item()
                epoch_loss /= batch_step
                loss1_current_epoch /= batch_step
                loss2_current_epoch /= batch_step
                if phase == 'val':
                    val_score /= batch_step
                    if self.lr_scheduler:
                        # lr_scheduler.step(epoch_loss)
                        self.lr_scheduler.step(val_score)

                    self.val_loss_list.append(epoch_loss)
                    self.val_loss_list_1.append(loss1_current_epoch)
                    self.val_loss_list_2.append(loss2_current_epoch)
                    print("Val loss on epoch %i: %f" % (epoch, epoch_loss))
                    print("Val score on epoch %i: %f" % (epoch, val_score))

                    file.write((f"Val loss on epoch {epoch}: {epoch_loss}"))
                    file.write((f"Val score on epoch {epoch}: {val_score}"))

                    file.write("\n")
                    if val_score >= self.best_val_score:
                        self.best_val_score = val_score
                        print("saving best model")
                        file.write("saving best model")
                        file.write("\n")
                        self.best_loss = epoch_loss
                        self.best_model = copy.deepcopy(
                            self.model.state_dict())
                        model_name = 'epoch{}.pt'.format(epoch)
                        save_dir = os.path.join(
                            self.output_save_dir, 'models/')
                        os.makedirs(save_dir, exist_ok=True)
                        torch.save(self.best_model, os.path.join(
                            save_dir, model_name))

                else:
                    std_1 = torch.exp(log_var_task1)**0.5
                    std_2 = torch.exp(log_var_task2)**0.5
                    print([std_1.item(), std_2.item()])
                    self.train_loss_list.append(epoch_loss)
                    self.train_loss_list_1.append(loss1_current_epoch)
                    self.train_loss_list_2.append(loss2_current_epoch)
                    print("Train loss on epoch %i: %f" % (epoch, epoch_loss))
                    file.write((f"Train loss on epoch {epoch}: {epoch_loss}"))
                    file.write("\n")

            torch.save(self.model.state_dict(), os.path.join(
                save_dir, 'last_epoch.pt'))

            time_elapsed = time.time() - since
            print('{:.0f}m {:.0f}s\n'.format(
                time_elapsed // 60, time_elapsed % 60))
            file.write('{:.0f}m {:.0f}s\n'.format(
                time_elapsed // 60, time_elapsed % 60))
            file.write("\n")

        print('Best val loss: {:4f}'.format(self.best_loss))
        print('Best val score: {:4f}'.format(self.best_val_score))

        file.write('Best val loss: {:4f}'.format(self.best_loss))
        file.write('Best val score: {:4f}'.format(self.best_val_score))

        file.write("\n")
        file.close()
        # load best model weights
        self.plot_loss_functions('total')

        return self.model
