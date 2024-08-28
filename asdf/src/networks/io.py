# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

import os
import importlib

import torch

from asdf.src.settings.settings import Settings
import asdf.src.misc.fileutils as fileutils
from collections import OrderedDict
new_state_dict = OrderedDict()

def load_network(epoch: int, device):
    model_filepath = os.path.join(fileutils.get_network_folder(), 'epoch.{}.pt'.format(epoch))
    loaded_states = torch.load(model_filepath, map_location=device)
    state_dict = loaded_states['model_state_dict']
    if next(iter(state_dict)).startswith('module.'):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            key = k[7:]  # remove `module.`
            new_state_dict[key] = v
        state_dict = new_state_dict
    net = initialize_net()
    net.to(device)
    net.load_state_dict(state_dict)
    return net

def save_state(filename, epoch, net, optimizer):
    model_dict = {'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
    filename = fileutils.ensure_ext('{}.{}'.format(fileutils.remove_ext(filename, '.pt'), epoch), '.pt')
    torch.save(model_dict, filename)
    print('Model saved to: {}'.format(filename))

def load_state(filename, epoch, net, optimizer, device):
    filename = fileutils.ensure_ext('{}.{}'.format(fileutils.remove_ext(filename, '.pt'), epoch), '.pt')
    loaded_states = torch.load(filename, map_location=device)
    net.load_state_dict(loaded_states['model_state_dict'])
    optimizer.load_state_dict(loaded_states['optimizer_state_dict'])

# This allows to select the network class by using the class name in Settpings
def initialize_net():
    module, class_name = Settings().network.network_class.rsplit('.', 1)
    FooBar = getattr(importlib.import_module(module), class_name)
    return FooBar()
