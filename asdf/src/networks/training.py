# Copyright 2020, 2024 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

import os
import time
import sys
import builtins

import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import asdf.src.networks.io as network_io
from asdf.src.settings.settings import Settings
import asdf.src.misc.fileutils as fileutils


def train_network(trainLoader: DataLoader, resume_epoch: int = 0):

    settings = Settings().network

    net = network_io.initialize_net()
    net.to(Settings().computing.device)
    net_module = net
    gpu_id = Settings().computing.local_gpu_id
    if Settings().computing.world_size > 1:
        net = DistributedDataParallel(net, device_ids=[gpu_id], output_device=gpu_id, find_unused_parameters=True)
        net_module = net.module
        
    #print_learnable_parameters(net)

    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print_once('Number of trainable parameters: {}'.format(total_params))

    # weight = torch.FloatTensor([0.1, 0.9]).to(Settings().computing.device)
    # criterion = nn.CrossEntropyLoss(weight=weight)
    criterion = nn.CrossEntropyLoss()

    optimizer = _init_optimizer(net, settings)

    log_folder = fileutils.get_network_log_folder()
    network_folder = fileutils.get_network_folder()
    output_filename = os.path.join(network_folder, 'epoch')

    if resume_epoch < 0:
        resume_epoch = -resume_epoch
        print_once('Computing metrics for epoch {}...'.format(resume_epoch))
        network_io.load_state(output_filename, resume_epoch, net, optimizer, Settings().computing.device)
        return net, True, resume_epoch
    elif resume_epoch > 0:
        print_once('Resuming network training from epoch {}...'.format(resume_epoch))
        network_io.load_state(output_filename, resume_epoch, net, optimizer, Settings().computing.device)

    net.train()

    for epoch in range(1, settings.epochs_per_train_call + 1):

        if Settings().computing.world_size > 1:
            trainLoader.sampler.set_epoch(epoch + resume_epoch) # This is needed for correct shuffling with DistributedSampler (see the PyTorch docs)

        current_learning_rate = optimizer.param_groups[0]['lr']

        logfilename = os.path.join(log_folder, 'gpu{}_epoch.{}.log'.format(Settings().computing.global_process_rank, epoch + resume_epoch))
        logfile = open(logfilename, 'w')
        print_once('GPU {}: Log file created: {}'.format(gpu_id, logfilename))

        training_loss = 0
        optimizer.zero_grad()

        if Settings().computing.world_size > 1:
            torch.distributed.barrier()  # Sync processes before timing for nicer outputs
        start_time = time.time()

        losses = []
        print_once('GPU {}: Iterating over training minibatches...'.format(gpu_id))

        for i, (batch_x, batch_y) in enumerate(trainLoader):

            # Copying audio_utils to GPU:
            batch_x = batch_x.to(Settings().computing.device)
            batch_y = batch_y.view(-1).type(torch.int64).to(Settings().computing.device)

            _, batch_out = net(batch_x)

            loss = (criterion(batch_out, batch_y)) / settings.optimizer_step_interval
            loss.backward()

            # Updating weights:
            if i % settings.optimizer_step_interval == settings.optimizer_step_interval - 1:
                optimizer.step()
                optimizer.zero_grad()


            minibatch_loss = loss.item() * settings.optimizer_step_interval
            losses.append(minibatch_loss)
            training_loss += minibatch_loss

            # Computing train and test accuracies and printing status:
            if i % settings.print_interval == settings.print_interval - 1:
                output = 'GPU {}: Epoch {}, Time: {:.0f} s, Batch {}/{}, lr: {:.6f}, train-loss: {:.3f} '.format(gpu_id, epoch + resume_epoch, time.time() - start_time, i + 1, len(trainLoader), current_learning_rate, training_loss / settings.print_interval)
                print_once(output)
                logfile.write(output + '\n')
                training_loss = 0

        # current_loss = np.asarray(losses).mean()
        current_loss = torch.mean(torch.tensor(losses)).to(Settings().computing.device)
        if (Settings().computing.world_size > 1):  # Broadcast decision to all processes
            torch.distributed.reduce(current_loss, 0, op=torch.distributed.ReduceOp.AVG)
            # print(current_loss)

        # Learning rate update:
        update_flag = torch.zeros(1).to(Settings().computing.device)
        if Settings().computing.global_process_rank == 0:  # Decision made based on the loss of the first process
            prev_loss = net_module.training_loss.item()
            current_loss = current_loss.item()
            room_for_improvement = max(Settings().network.min_room_for_improvement, prev_loss - Settings().network.target_loss)
            loss_change = (prev_loss - current_loss) / room_for_improvement
            print('GPU {}: Average training loss reduced {:.2f}% from the previous epoch.'.format(gpu_id, loss_change*100))
            if loss_change < Settings().network.min_loss_change_ratio:
                update_flag[0] = 1
                print('GPU {}: Because loss change {:.2f}% <= {:.2f}%, the learning rate is lowered: {} --> {}'.format(gpu_id, loss_change*100, Settings().network.min_loss_change_ratio*100, optimizer.param_groups[0]['lr'], optimizer.param_groups[0]['lr'] * Settings().network.lr_update_ratio))
                print('GPU {}: Consecutive LR updates: {}'.format(gpu_id, net_module.consecutive_lr_updates[0] + 1))

            net_module.training_loss[0] = current_loss

        if Settings().computing.world_size > 1:  # Broadcast decision to all processes
            torch.distributed.broadcast(update_flag, 0)

        if update_flag[0] == 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * Settings().network.lr_update_ratio
            net_module.consecutive_lr_updates[0] += 1
        else:
            net_module.consecutive_lr_updates[0] = 0

        if Settings().computing.global_process_rank == 0:
            network_io.save_state(output_filename, epoch + resume_epoch, net, optimizer)

        if net_module.consecutive_lr_updates[0] >= Settings().network.max_consecutive_lr_updates:
            print_once('GPU {}: Stopping training because reached {} consecutive LR updates!'.format(gpu_id, Settings().network.max_consecutive_lr_updates))
            return net_module, True, epoch + resume_epoch

        logfile.close()

    return net_module, False, epoch + resume_epoch


def _init_optimizer(net, settings):
    params = get_weight_decay_param_groups(net, settings.weight_decay_skiplist)
    if settings.optimizer == 'sgd':
        return optim.SGD(params, lr=settings.initial_learning_rate, weight_decay=settings.weight_decay, momentum=settings.momentum)
    if settings.optimizer == 'adam':
        return optim.Adam(params, lr=settings.initial_learning_rate, weight_decay=settings.weight_decay)
    sys.exit('Unsupported optimizer: {}'.format(settings.optimizer))

def get_weight_decay_param_groups(model, skip_list):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if builtins.any(x in name for x in skip_list):
            no_decay.append(param)
            print_once('No weight decay applied to {}'.format(name))
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': Settings().network.weight_decay}]

def print_learnable_parameters(model: torch.nn.Module):
    if Settings().computing.global_process_rank == 0:
        print('Learnable parameters of the model:')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.numel())

def print_once(string: str):
    if Settings().computing.global_process_rank == 0:
        print(string)


