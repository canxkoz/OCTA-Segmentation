import copy
import os
import time
from tqdm import tqdm
import torch
from loss import calc_loss, MultitaskUncertaintyLoss
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
import math

class Trainer():
    def __init__(self, model, model_type, dtype, device, output_save_dir, dataloaders, batch_size, optimizer, class_weights,class_loss_function,
        reg_loss_function,patience, num_epochs, accuracy_metric,  lr_scheduler=None, start_epoch=1):
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
        self.best_val_loss = 1000
        self.batch_size = batch_size
        self.output_save_dir = output_save_dir
        self.dtype = dtype
        self.device = device
        
        self.class_loss_function = class_loss_function
        
        self.reg_loss_function = reg_loss_function
        
        self.accuracy_metric = accuracy_metric
        self.train_loss_list = []
        self.val_loss_list = []
        self.model_type = model_type

        self.train_loss_list_1 = []
        self.val_loss_list_1 = []

        self.train_loss_list_2 = []
        self.val_loss_list_2 = []
        
        self.std_1_list = []
        self.std_2_list = []


        self.bce_loss = []
        self.mse_loss = []
        
        
        self.class_weights = torch.tensor(class_weights).to(device)
        
        print('class weights used : !!!',self.class_weights)


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
            
            
            with open(os.path.join(self.output_save_dir, "Multi-loss_results.txt"), "w") as f:
                for epoch in range(len(self.std_1_list)):
                    f.write(f"--- epoch {epoch+1} ---\n")
                    f.write(f"Train Bce loss: {self.train_loss_list_1[epoch]}\n")
                    f.write(f"Train Mse loss: {self.train_loss_list_2[epoch]}\n")
                    f.write(f"Train Standard deviation 1: {self.std_1_list[epoch]}\n")
                    f.write(f"Train Standard deviation 2: {self.std_2_list[epoch]}\n")
            f.close()

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
            self.single_train()

        elif self.model_type == 'single_dropout':
            self.single_train()
        
        elif self.model_type == 'multi_task':
            print('multi')
            self.multi_task_train()

        elif self.model_type == 'multi_alpha':
            print('multi')
            self.multi_task_alpha_train()
        
        else:
            raise ValueError('Invalid model_type "%s"' % self.model_type)
        
    def multi_task_alpha_train(self):
        
        
        
        if not os.path.exists(self.output_save_dir):
            os.mkdir(self.output_save_dir)
        log_file = os.path.join(self.output_save_dir, "logs.txt")

        file = open(log_file, 'a')

        total_memory = f'{torch.cuda.get_device_properties(0).total_memory/ 1E9 if torch.cuda.is_available() else 0:.3g}G'

        log_var_task1 = torch.zeros((1,), requires_grad=True)
        log_var_task2 = torch.zeros((1,), requires_grad=True)
        params = [p for p in self.model.parameters()]
        #params = ([p for p in self.model.parameters()] +
                  #[log_var_task1] + [log_var_task2])
        #loss_combiner = MultitaskUncertaintyLoss()

        self.optimizer = optim.Adam(params, lr=1e-4,)
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.4, patience=5, min_lr=5e-6)
        
  
       
        for epoch in range(self.start_epoch, self.num_epochs+1):


            
            if epoch != 1:
                alpha = (sum(self.bce_loss) / len(self.bce_loss))/(sum(self.mse_loss) / len(self.mse_loss))
                #alpha = float(alpha.item())
                
                print('div factor:,',sum(self.bce_loss)/sum(self.mse_loss))

            self.bce_loss = []
            self.mse_loss = []
            
            print('Epoch {}/{}'.format(epoch, self.num_epochs))
            print('-' * 10)
            file.write('Epoch {}/{}'.format(epoch, self.num_epochs))
            file.write("\n")
            file.write('-' * 10)
            file.write("\n")
            since = time.time()
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                epoch_loss = 0.0
                loss1_current_epoch = 0
                loss2_current_epoch = 0
                val_loss = 0.0
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
                    #for inputs, label_mask, label_fdmap in tbar:
                    for inputs, label_mask, label_dist in tbar:

                        batch_step += 1
                        inputs = inputs.to(self.device).type(self.dtype)
                        label_mask = label_mask.to(
                            self.device).type(self.dtype)
                        label_dist = label_dist.to(
                            self.device).type(self.dtype)

                        # zero the parameter gradients
                        self.optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            output_mask, output_dist = self.model(inputs)

                            # binary classification
                            loss1 = calc_loss(output_mask, label_mask, class_weights=self.class_weights, loss_type=self.class_loss_function)
                            
                            
                           

                            # regression
                            amin = torch.amin(output_dist, dim=(-2, -1))
                            output_dist = torch.sub(output_dist, amin.reshape(
                                (amin.shape[0], amin.shape[1], 1, 1)))
                            loss2 = calc_loss(output_dist, label_dist,
                                              loss_type=self.reg_loss_function)
                            
                                                       
                            
                            print('epoch is',epoch)
                            if epoch == 1:

                                alpha = 0
                             
                            print('alpha is:',alpha)

                            print('len of bce list ist ',len(self.bce_loss))
                            loss = loss1 + alpha*loss2
                            
                            
                        
                            
                          

                            loss = loss.to(self.device)
                            reserved = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                            mem = reserved + '/' + total_memory
                            # backward + optimize only if in training phase
                            if phase == 'train':
                                
                                print('adding loss1',loss1.item())

                                self.bce_loss.append(loss1.item())
                                print('adding loss2',loss2.item())
                            
                                self.mse_loss.append(loss2.item())
                                
                                loss.backward()
                                self.optimizer.step()
                                epoch_loss += loss.item()
                                tbar.set_postfix(
                                    loss=epoch_loss/batch_step,  memory=mem)
                                loss1_current_epoch += loss1.detach().item()
                                loss2_current_epoch += loss2.detach().item()
                                
                            else:
                                epoch_loss += loss.item()
                                val_loss += calc_loss(output_mask, label_mask,
                                                       loss_type=self.accuracy_metric)
                                tbar.set_postfix(loss=epoch_loss/batch_step,
                                                 accuracy=(val_loss.item()/(batch_step)), memory=mem)
                                loss1_current_epoch += loss1.detach().item()
                                loss2_current_epoch += loss2.detach().item()
                epoch_loss /= batch_step
                loss1_current_epoch /= batch_step
                loss2_current_epoch /= batch_step
                if phase == 'val':
                    val_loss /= batch_step
                    if self.lr_scheduler:
                        # lr_scheduler.step(epoch_loss)
                        self.lr_scheduler.step(val_loss)

                    self.val_loss_list.append(epoch_loss)
                    self.val_loss_list_1.append(loss1_current_epoch)
                    self.val_loss_list_2.append(loss2_current_epoch)
                    print("Val loss on epoch %i: %f" % (epoch, epoch_loss))
                    print("Val score on epoch %i: %f" % (epoch, val_loss))

                    file.write((f"Val loss on epoch {epoch}: {epoch_loss}"))
                    file.write((f"Val score on epoch {epoch}: {val_loss}"))

                    file.write("\n")
                    if val_loss <= self.best_val_loss:
                        self.best_val_loss = val_loss
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
                    self.std_1_list.append(std_1.item())
                    self.std_2_list.append(std_2.item())
                    #print([std_1.item(), std_2.item()])
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
        
        plt.plot(range(1, self.num_epochs + 1), self.std_1_list, label="std_BCE")
        plt.plot(range(1, self.num_epochs + 1), self.std_2_list, label="std_MSE")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Values of std_bce and std_mse over epochs")
        plt.legend()
        plt.savefig("std_plot.png")

        std_1_arr = np.array(self.std_1_list)
        std_2_arr = np.array(self.std_2_list)

        #np.savetxt('std_values.txt', np.column_stack((std_1_arr, std_2_arr)), delimiter='\t')

        print('Best val loss: {:4f}'.format(self.best_loss))
        print('Best val score: {:4f}'.format(self.best_val_loss))

        file.write('Best val loss: {:4f}'.format(self.best_loss))
        file.write('Best val score: {:4f}'.format(self.best_val_loss))

        file.write("\n")
        file.close()
        # load best model weights
        self.plot_loss_functions('total')

        return self.model


    def multi_task_train(self):
        
        
        
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
            self.optimizer, mode='min', factor=0.5, patience=30, min_lr=5e-6)
        
        

        for epoch in range(self.start_epoch, self.num_epochs+1):
            print('Epoch {}/{}'.format(epoch, self.num_epochs))
            print('-' * 10)
            file.write('Epoch {}/{}'.format(epoch, self.num_epochs))
            file.write("\n")
            file.write('-' * 10)
            file.write("\n")
            since = time.time()
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                epoch_loss = 0.0
                loss1_current_epoch = 0
                loss2_current_epoch = 0
                val_loss = 0.0
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
                    #for inputs, label_mask, label_fdmap in tbar:
                    for inputs, label_mask, label_dist in tbar:

                        batch_step += 1
                        inputs = inputs.to(self.device).type(self.dtype)
                        label_mask = label_mask.to(
                            self.device).type(self.dtype)
                        label_dist = label_dist.to(
                            self.device).type(self.dtype)

                        # zero the parameter gradients
                        self.optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            output_mask, output_dist = self.model(inputs)

                            # binary classification
                            loss1 = calc_loss(output_mask, label_mask, class_weights=self.class_weights, loss_type=self.class_loss_function)
                           

                            # regression
                            amin = torch.amin(output_dist, dim=(-2, -1))
                            output_dist = torch.sub(output_dist, amin.reshape(
                                (amin.shape[0], amin.shape[1], 1, 1)))
                            loss2 = calc_loss(output_dist, label_dist,
                                              loss_type=self.reg_loss_function)
                            
                           

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
                                val_loss += calc_loss(output_mask, label_mask,
                                                       loss_type=self.accuracy_metric)
                                tbar.set_postfix(loss=epoch_loss/batch_step,
                                                 accuracy=(val_loss.item()/(batch_step)), memory=mem)
                                loss1_current_epoch += loss1.detach().item()
                                loss2_current_epoch += loss2.detach().item()
                epoch_loss /= batch_step
                loss1_current_epoch /= batch_step
                loss2_current_epoch /= batch_step
                if phase == 'val':
                    val_loss /= batch_step
                    if self.lr_scheduler:
                        # lr_scheduler.step(epoch_loss)
                        self.lr_scheduler.step(val_loss)

                    self.val_loss_list.append(epoch_loss)
                    self.val_loss_list_1.append(loss1_current_epoch)
                    self.val_loss_list_2.append(loss2_current_epoch)
                    print("Val loss on epoch %i: %f" % (epoch, epoch_loss))
                    print("Val score on epoch %i: %f" % (epoch, val_loss))

                    file.write((f"Val loss on epoch {epoch}: {epoch_loss}"))
                    file.write((f"Val score on epoch {epoch}: {val_loss}"))

                    file.write("\n")
                    if val_loss <= self.best_val_loss:
                        self.best_val_loss = val_loss
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
                    self.std_1_list.append(std_1.item())
                    self.std_2_list.append(std_2.item())
                    #print([std_1.item(), std_2.item()])
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
        
        plt.plot(range(1, self.num_epochs + 1), self.std_1_list, label="std_1")
        plt.plot(range(1, self.num_epochs + 1), self.std_2_list, label="std_2")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Values of std_1 and std_2 over epochs")
        plt.legend()

        plt.savefig(os.path.join(self.output_save_dir,"std_plot.png"))

        std_1_arr = np.array(self.std_1_list)
        std_2_arr = np.array(self.std_2_list)

        #np.savetxt('std_values.txt', np.column_stack((std_1_arr, std_2_arr)), delimiter='\t')

        print('Best val loss: {:4f}'.format(self.best_loss))
        print('Best val score: {:4f}'.format(self.best_val_loss))

        file.write('Best val loss: {:4f}'.format(self.best_loss))
        file.write('Best val score: {:4f}'.format(self.best_val_loss))

        file.write("\n")
        file.close()
        # load best model weights
        self.plot_loss_functions('total')

        return self.model

    def single_train(self):
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
                val_loss = 0.0
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
                            loss = calc_loss(output_mask, label_mask, class_weights=self.class_weights, loss_type=self.class_loss_function)                 

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                self.optimizer.step()
                                epoch_loss += loss.item()
                                tbar.set_postfix(loss=epoch_loss/batch_step)

                            else:
                                epoch_loss += loss.item()
                                val_loss += calc_loss(output_mask, label_mask,
                                                    loss_type=self.accuracy_metric)
                                tbar.set_postfix(loss=epoch_loss/batch_step,
                                                 accuracy=(val_loss/(batch_step)))

                epoch_loss /= batch_step
                # deep copy the model
                if phase == 'val':
                    val_loss /= batch_step
                    if self.lr_scheduler:
                        # lr_scheduler.step(epoch_loss)
                        self.lr_scheduler.step(val_loss)

                    self.val_loss_list.append(epoch_loss)
                    print("Val loss on epoch %i: %f" % (epoch, epoch_loss))
                    print("Val score on epoch %i: %f" % (epoch, val_loss))

                    file.write((f"Val loss on epoch {epoch}: {epoch_loss}"))
                    file.write((f"Val score on epoch {epoch}: {val_loss}"))

                    file.write("\n")
                    if val_loss <= self.best_val_loss:
                        self.best_val_loss = val_loss
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
                    
                    
            print(save_dir)

            torch.save(self.model.state_dict(), os.path.join(save_dir, 'last_epoch.pt'))

            time_elapsed = time.time() - since
            print('{:.0f}m {:.0f}s\n'.format(
                time_elapsed // 60, time_elapsed % 60))
            file.write('{:.0f}m {:.0f}s\n'.format(
                time_elapsed // 60, time_elapsed % 60))
            file.write("\n")

        print('Best val loss: {:4f}'.format(self.best_loss))
        print('Best val score: {:4f}'.format(self.best_val_loss))

        file.write('Best val loss: {:4f}'.format(self.best_loss))
        file.write('Best val score: {:4f}'.format(self.best_val_loss))

        file.write("\n")
        file.close()
        # load best model weights
        self.model.load_state_dict(self.best_model)
        self.plot_loss_functions('total')

        return self.model
