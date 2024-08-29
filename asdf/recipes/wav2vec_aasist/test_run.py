# Copyright 2024 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).
# Main script for wav2vec_aasist recipe.

import sys
import os

# Adding the project root to the path to make imports to work regardless from where this file was executed:
sys.path.append(os.path.dirname(os.path.abspath(__file__)).rsplit('asdf', 1)[0])
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["NUMEXPR_NUM_THREADS"] = '1'
os.environ["OMP_NUM_THREADS"] = '1'
from dataclasses import dataclass
from typing import Union, Tuple

import torch.distributed

from datetime import datetime

from asdf.src.settings.abstract_settings import AbstractSettings
from asdf.src.settings.settings import Settings
import asdf.src.misc.fileutils as fileutils
from asdf.src.misc.miscutils import dual_print
import asdf.src.networks.training as aasist_trainer
from asdf.src.networks import io
import asdf.src.misc.recipeutils as recipeutils
import asdf.src.networks.dataloaders as dataloaders
from asdf.src.evaluation.eval_metrics import compute_eer_from_score_file

import socket
import subprocess

# os.environ["NCCL_DEBUG"] = "INFO"



hostname = socket.gethostname()
print("Hostname: " + hostname)
# Host name can be used to configure database folders for each server you are using

trainListFolder = os.path.join(fileutils.get_folder_of_file(__file__), 'training_lists') + '/'
evalListFolder = os.path.join(fileutils.get_folder_of_file(__file__), 'eval_lists') + '/'

# TODO UPDATE the folders below to match yours (no need to update if you do not use the datasets below):
#databaseFolder = os.environ.get('LOCAL_SCRATCH') + '/'
databaseFolder = '/data/vvestman/'

training_datasets = {
    'asvspoof19': (databaseFolder + 'LA', trainListFolder + 'asvspoof19_la_train_list.txt'),
    'mlaad':  (databaseFolder + 'MLAADv3_16khz', trainListFolder + 'mlaad_train_list.txt'),
    'm-ailabs': (databaseFolder + 'm-ailabs-mlaad-sources', trainListFolder + 'm_ailabs_train_list.txt'),
    'wavefake': (databaseFolder + 'WaveFake_16khz', trainListFolder + 'wavefake_train_list.txt'),
    'itw': (databaseFolder + 'in_the_wild', trainListFolder + 'in_the_wild_train_list.txt'),
    'for': (databaseFolder + 'for-norm', trainListFolder + 'for_norm_train_list.txt'),
}

evaluation_lists = {
    'asvspoof19-dev': (databaseFolder + 'LA', evalListFolder + 'asvspoof19_dev_trials.txt'),
    'asvspoof19-eval': (databaseFolder + 'LA', evalListFolder + 'asvspoof19_eval_trials.txt'),
    'asvspoof21-df-progress': (databaseFolder + 'asvspoof2021', evalListFolder + 'asvspoof21_df_progress_trials.txt'),
    'asvspoof21-df-eval': (databaseFolder + 'asvspoof2021', evalListFolder + 'asvspoof21_df_eval_trials.txt'),
    'asvspoof21-la-progress': (databaseFolder + 'asvspoof2021', evalListFolder + 'asvspoof21_la_progress_trials.txt'),
    'asvspoof21-la-eval': (databaseFolder + 'asvspoof2021', evalListFolder + 'asvspoof21_la_eval_trials.txt'),
    'for-validation': (databaseFolder + 'for-norm', evalListFolder + 'for_norm_validation_trials.txt'),
    'for-testing': (databaseFolder + 'for-norm', evalListFolder + 'for_norm_testing_trials.txt'),
}


eer = compute_eer_from_score_file(score_file='/home/ville/scores_epoch_7_asvspoof19_dev_trials.txt')
print(eer)