# Copyright 2020, 2024 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

from typing import NamedTuple, Iterable, List
import os
import sys

import torch.distributed

from asdf.src.settings.settings import Settings
import asdf.src.misc.fileutils as fileutils


def parse_recipe_arguments_and_set_up_distributed_computing(arguments : List[str]) -> List[str]:

    global_rank = os.environ.get('RANK')
    local_rank = os.environ.get('LOCAL_RANK')
    world_size = os.environ.get('WORLD_SIZE')
    local_world_size = os.environ.get('LOCAL_WORLD_SIZE')

    if global_rank is not None and local_rank is not None and world_size is not None and local_world_size is not None:
        Settings().computing.world_size = int(world_size)
        Settings().computing.local_process_rank = int(local_rank)
        Settings().computing.global_process_rank = int(global_rank)
        torch.distributed.init_process_group('nccl')
    else:
        Settings().computing.world_size = 1
        Settings().computing.local_process_rank = 0
        Settings().computing.global_process_rank = 0

    if  Settings().computing.world_size == 1:
        if Settings().computing.use_gpu:
            print("Using single GPU")
        else:
            print("Using CPU")
    else:
        print("Using {} GPUs spread over {} nodes".format(Settings().computing.world_size,  Settings().computing.world_size // int(local_world_size)))

    # distributed = True
    arguments = arguments[1:]
    if not arguments:
        sys.exit('Give one or more run configs as argument(s)!')

    return arguments


def find_last_epoch():
    epoch = 1
    while os.path.exists(os.path.join(fileutils.get_network_folder(), 'epoch.{}.pt'.format(epoch))):
        epoch += 1
    if epoch == 1:
        sys.exit('ERROR: trying to load model that has not been trained yet ({})'.format(os.path.join(fileutils.get_network_folder(), 'epoch.{}.pt'.format(epoch))))
    return epoch - 1



