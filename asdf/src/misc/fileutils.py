# Copyright 2020 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).

import os
from os.path import isfile, join
import datetime
from typing import List
import time

from asdf.src.settings.settings import Settings

# if you run code in a distributed system where the nodes do not have access to the same storage, this needs to be modified somehow (you'll probably run into errors)
# --> use local_process_rank instead of global
def ensure_exists(folder: str):
    if not os.path.exists(folder):
        if Settings().computing.global_process_rank == 0:
            os.makedirs(folder)
        else:
            i = 0
            while not os.path.exists(folder):
                i += 1
                time.sleep(2)
                if i == 10: # Give up after 20 secs
                    break

def ensure_ext(filename: str, ext: str) -> str:
    if not ext.startswith('.'):
        ext = '.' + ext
    if not filename.endswith(ext):
        filename = filename + ext
    return filename

def remove_ext(filename: str, ext: str) -> str:
    if not ext.startswith('.'):
        ext = '.' + ext
    if filename.endswith(ext):
        filename = filename[:-len(ext)]
    return filename


def get_network_folder() -> str:
    folder = os.path.join(Settings().paths.output_folder, Settings().paths.system_folder, 'networks')
    ensure_exists(folder)
    return folder

def get_network_log_folder() -> str:
    folder = os.path.join(Settings().paths.output_folder, Settings().paths.system_folder, 'networks', 'logs')
    ensure_exists(folder)
    return folder

def get_score_folder() -> str:
    folder = os.path.join(Settings().paths.output_folder, Settings().paths.system_folder, 'scores')
    ensure_exists(folder)
    return folder

def get_post_sad_lengths_folder() -> str:
    folder = os.path.join(Settings().paths.output_folder, Settings().paths.sad_folder, 'post_sad_lengths')
    ensure_exists(folder)
    return folder

def get_score_output_file(score_file_name: str , prefix: str = None) -> str:
    if prefix:
        return os.path.join(get_score_folder(), 'scores_{}_{}'.format(prefix, score_file_name))
    return os.path.join(get_score_folder(), 'scores_{}'.format(score_file_name))

def get_results_folder() -> str:
    folder = os.path.join(Settings().paths.output_folder, Settings().paths.system_folder, 'results')
    ensure_exists(folder)
    return folder

def get_new_results_file() -> str:
    results_folder = get_results_folder()
    return os.path.join(results_folder, 'results_{}.txt'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

def get_folder_of_file(filename: str) -> str:
    return os.path.dirname(os.path.abspath(filename))
