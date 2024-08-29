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
databaseFolder = os.environ.get('LOCAL_SCRATCH') + '/'
#databaseFolder = '/data/vvestman/'

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


# Initializing settings:
Settings(os.path.join(fileutils.get_folder_of_file(__file__), 'configs', 'init_config.py'))

# Get full path of run config file:
run_config_file = os.path.join(fileutils.get_folder_of_file(__file__), 'configs', 'run_configs.py')

# Get run configs from command line arguments and set up distributed computing
run_configs = recipeutils.parse_recipe_arguments_and_set_up_distributed_computing(sys.argv)

if Settings().computing.global_process_rank == 0:
    # Downloading pretrained Wav2Vec2 model
    url = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)).rsplit('asdf', 1)[0], 'asdf', 'src', 'networks')
    subprocess.run(["wget", "-nc", "-P", path, url])

    Settings().print()


# config loop:
for settings_string in Settings().load_settings(run_config_file, run_configs):

    # Common preparation:
    trialLists = {}
    for key in Settings().recipe.evaluation_sets:
        dataset_folder, list_path = evaluation_lists[key]
        trialList = dataloaders.TrialList(dataset_folder=dataset_folder, trial_list_file=list_path)
        trialLists[key] = trialList


    # Predetermine utterance lengths (for faster scoring), stage 0
    if Settings().recipe.execute_stage(0, multigpu_supported=True):
        for key in trialLists:
            trialLists[key].compute_post_sad_lengths()


    # Network training, stage 1
    if Settings().recipe.execute_stage(1, multigpu_supported=True):

        for key in trialLists:
            trialLists[key].init_data_loader()

        train_sets = []
        for train_set_id in Settings().recipe.training_sets:
            train_sets.append(training_datasets[train_set_id])


        trainLoader = dataloaders.get_train_loader(train_sets)

        if Settings().computing.global_process_rank == 0:
            result_file = open(fileutils.get_new_results_file(), 'w')
            result_file.write(settings_string + '\n\n')

        for epoch in range(Settings().network.resume_epoch, Settings().network.max_epochs,
                           Settings().network.epochs_per_train_call):

            network, stop_flag, epoch = aasist_trainer.train_network(trainLoader, epoch)

            output_text = "Scores - Epoch {} [{}]:\n".format(epoch, datetime.now())
            for key in trialLists:
                score_file = trialLists[key].score(network, 'epoch_{}'.format(epoch))
                if Settings().computing.global_process_rank == 0:
                    eer = compute_eer_from_score_file(score_file=score_file)
                    output_text += '{} EER = {:.4f}'.format(key, eer)
                    dual_print(result_file, output_text)
                    output_text = ''

            network = None
            torch.cuda.empty_cache()
            if stop_flag:
                break

        if Settings().computing.global_process_rank == 0:
            result_file.close()



    # Network evaluation, stage 2
    if Settings().recipe.execute_stage(2, multigpu_supported=True):

        for key in trialLists:
            trialLists[key].init_data_loader()

        epochs = Settings().recipe.selected_epochs if Settings().recipe.selected_epochs else (recipeutils.find_last_epoch(),)
        if Settings().computing.global_process_rank == 0:
            result_file = open(fileutils.get_new_results_file(), 'w')
            result_file.write(settings_string + '\n\n')
        for epoch in epochs:
            network = io.load_network(epoch, Settings().computing.device)

            output_text = "Scores - Epoch {}:\n".format(epoch)
            for key in trialLists:
                score_file = trialLists[key].score(network, 'epoch_{}'.format(epoch))

                if Settings().computing.global_process_rank == 0:
                    eer = compute_eer_from_score_file(score_file=score_file)

                if Settings().computing.global_process_rank == 0:
                    output_text += '{} EER = {:.4f}'.format(key, eer)
                    dual_print(result_file, output_text)
                    output_text = ''

        if Settings().computing.global_process_rank == 0:
            result_file.close()

print('Process {}: All done!'.format(Settings().computing.global_process_rank))
