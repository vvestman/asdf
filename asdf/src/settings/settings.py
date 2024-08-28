# Copyright 2024 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

import sys
from dataclasses import dataclass, field
from typing import Tuple, Dict, List

import torch
import torch.cuda

from asdf.src.settings.abstract_settings import AbstractSettings

# Edit the configs below using separate (recipe specific) settings files unless you want to the change the default values for all systems.
# New settings can be added freely.

@dataclass
class PathSettings(AbstractSettings):
    output_folder: str = '/all/the/system/outputs/go/here/'
    system_folder: str = 'system1'  # Relative folder for antispoofing system (contains a networks, output scores, etc...)
    sad_folder: str = 'sad' # Relative folder for utterance lengths after SAD

@dataclass
class ComputingSettings(AbstractSettings):
    network_dataloader_workers: int = 10
    use_gpu: bool = True  # If false, then use CPU
    gpu_ids: Tuple[int] = (0,)  # Which GPUs to use?

    local_process_rank: int = 0  # Automatically updated by distributed computing scripts
    global_process_rank: int = 0  # Automatically updated by distributed computing scripts
    local_gpu_id: int = 0  # Automatically updated by distributed computing scripts
    world_size: int = 1  # How many processes in distributed computing? Automatically set.
    device: torch.device = field(init=False)  # Automatically set.


@dataclass
class NetworkSettings(AbstractSettings):
    resume_epoch: int = 0  # Continue training from the given epoch (0 = start training new model)
    max_epochs: int = 1000  # Stop training after this number of epochs (if not already stopped by other means)
    epochs_per_train_call: int = 1  # Increase this if you do not want to compute EER after every epoch

    minibatch_size: int = 24
    eval_minibatch_size_factor: float = 2 # Scale batch size by this when scoring trials (set this based on avail. GPU mem)

    train_clip_size: int = 16000  # Training example size in samples (16000 = 1 second, if fs=16kHz)

    min_evaluation_utterance_length_in_samples: int = 16000 # Min length of utterance in scoring
    max_evaluation_utterance_length_in_samples: int = 1000000 # Max length of utterance in scoring
    # GPU scoring algorithm organizes audio files with close to equal durations into minibatches by cutting out ends of the longest audio files
    max_cut_proportion: float = 0.05   # Maximum proportion of utterance to be cut out in scoring

    print_interval: int = 10  # How often to print tranining loss (in minibatches)
    scoring_print_interval: int = 50  # Print interval in scoring (in minibatches)

    optimizer: str = 'sgd'

    max_consecutive_lr_updates = 2
    lr_update_ratio = 0.5
    initial_learning_rate: float = 0.1
    min_loss_change_ratio: float = 0.01
    min_room_for_improvement: float = 0.1
    target_loss: float = 0.1

    momentum: float = 0  # (0 to disable)

    weight_decay: float = 0.001    # Weight decay for utterance-level layers
    weight_decay_skiplist: Tuple[str] = ('batchnorm',)

    optimizer_step_interval: int = 1  # This can be used to combine gradients of multiple minibatches before updating weights

    # Network architecture:
    network_class: str = 'asdf.src.networks.architectures.SSL_AASIST'



class RawBoostSettings(AbstractSettings):
    enabled: bool = False
    applyingRatio: float = 0.8
    algo: int = 5 # Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), 5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2)

    #LnL_convolotive_noise parameters:
    nBands: int = 5 # number of notch filters.The higher the number of bands, the more aggresive the distortions is.
    minF: int = 20 # minimum centre frequency [Hz] of notch filter.
    maxF: int = 8000 # maximum centre frequency [Hz] (<sr/2)  of notch filter.
    minBW: int = 100 # minimum width [Hz] of filter.
    maxBW: int = 1000 # maximum width [Hz] of filter.
    minCoeff: int = 10 # minimum filter coefficients. More the filter coefficients more ideal the filter slope.
    maxCoeff: int = 100 # maximum filter coefficients. More the filter coefficients more ideal the filter slope.
    minG: int = 0 # minimum gain factor of linear component.
    maxG: int = 0 # maximum gain factor of linear component.
    minBiasLinNonLin: int = 5 # minimum gain difference between linear and non-linear components.
    maxBiasLinNonLin: int = 20 # maximum gain difference between linear and non-linear components.
    N_f: int = 5 # order of the (non-)linearity where N_f=1 refers only to linear components.

    #ISD_additive_noise parameters:
    P: int = 10 # Maximum number of uniformly distributed samples in [%].
    g_sd: int = 2 # gain parameters > 0.

    #SSI_additive_noise_parameters:
    SNRmin: int = 10 # Minimum SNR value for coloured additive noise.
    SNRmax: int = 40 # Maximum SNR value for coloured additive noise.


class SadSettings(AbstractSettings):
    mode: str = 'off' #off/endpoint/on


@dataclass
class RecipeSettings(AbstractSettings):
    start_stage: int = 0
    end_stage: int = 100
    selected_epochs: Tuple[int] = None  # Find the last epoch automatically
    training_sets: Tuple[str] = ('asvspoof19',)
    evaluation_sets: Tuple[str] = ('asvspoof19-dev', 'asvspoof19-eval')

    def execute_stage(self, stage: int, multigpu_supported: bool = False) -> bool:
        return_true =  self.start_stage <= stage <= self.end_stage and (multigpu_supported or Settings().computing.global_process_rank == 0)
        if return_true and Settings().computing.world_size > 1:
            torch.distributed.barrier()  # Sync all procesess before entering to the stage
        return return_true





@dataclass
class Settings(AbstractSettings):
    init_settings_file: str = None
    paths: PathSettings = field(default_factory=lambda: PathSettings(), init=False)
    computing: ComputingSettings = field(default_factory=lambda: ComputingSettings(), init=False)
    network: NetworkSettings = field(default_factory=lambda: NetworkSettings(), init=False)
    sad: NetworkSettings = field(default_factory=lambda: SadSettings(), init=False)
    rawboost: RawBoostSettings = field(default_factory=lambda: RawBoostSettings(), init=False)
    recipe: RecipeSettings = field(default_factory=lambda: RecipeSettings(), init=False)

    def __post_init__(self):
        # Initial settings
        if self.init_settings_file is not None:
            self.set_initial_settings(self.init_settings_file)

    def post_update_call(self):
        # Set GPU device
        self.computing.device = torch.device("cpu")
        if torch.cuda.is_available() and self.computing.use_gpu:
            self._set_local_gpu_id_using_local_rank()
            self.computing.device = torch.device('cuda:{}'.format(self.computing.local_gpu_id))
            torch.cuda.set_device(self.computing.device)
            torch.backends.cudnn.benchmark = False
            print('Using GPU (gpu_id = {})!'.format(self.computing.local_gpu_id))
        else:
            print('Cuda is not available! Using CPU!')

    def _set_local_gpu_id_using_local_rank(self):
        if len(self.computing.gpu_ids) < self.computing.world_size: # not enough gpu_ids, so using identity mapping
            self.computing.local_gpu_id = self.computing.local_process_rank
        else:
            self.computing.local_gpu_id = self.computing.gpu_ids[self.computing.local_process_rank]
