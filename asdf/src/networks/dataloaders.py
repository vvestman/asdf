import os
import random
from math import ceil
from typing import List, Tuple

import numpy as np

import torch
import torchaudio
from torch import Tensor
from torch.distributed import all_gather, gather
from torch.utils.data import DataLoader, Dataset

from asdf.src.audio_utils import sad
from asdf.src.misc import fileutils
from asdf.src.settings.settings import Settings
import asdf.src.audio_utils.rawboost as rawboost
from asdf.src.audio_utils.padding import pad_random, tile_to_size
from torch.utils.data.distributed import DistributedSampler

def load_audio(path):
    x, fs = torchaudio.load(path)
    x = x.numpy().flatten()
    return x, fs

# DO NOT USE THIS SAD IMPLEMENTATION (MAKE A BETTER ONE)
def normalize_and_apply_sad(signal, fs, file):
    max_value = np.max(np.abs(signal))
    if max_value > 0:
        signal = signal / max_value
    if Settings().sad.mode == 'off':
        return signal
    E = sad.compute_sad_energies(signal, fs)
    output_signal, sad_signal, E = sad.endpoint_sad(signal, fs, E, -35, file)
    if Settings().sad.mode == 'endpoint':
        return output_signal
    output_signal2, sad_signal2, E = sad.sad(output_signal, fs, E, -35, 0.00, True, file)
    if Settings().sad.mode == 'on':
        return output_signal2

def get_train_loader(train_lists: List[Tuple[str, str]]):
    dataset = TrainDataset(train_lists)

    if Settings().computing.world_size > 1:
        sampler = DistributedSampler(dataset) # Sampler shuffles by default
        loader = DataLoader(dataset,
                            sampler=sampler,
                            batch_size=Settings().network.minibatch_size,
                            drop_last=True,
                            pin_memory=True,
                            num_workers=Settings().computing.network_dataloader_workers)
    else:
        loader = DataLoader(dataset,
                            batch_size=Settings().network.minibatch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            num_workers=Settings().computing.network_dataloader_workers)
    return loader


class TrainDataset(Dataset):
    def __init__(self, train_lists: List[Tuple[str, str]]):
        self.utterances = []

        for databasePath, trainList in train_lists:
            with open(trainList, 'r') as f:
                for line in f:
                    parts = line.split(' ')
                    utt = os.path.join(databasePath, parts[0])
                    label = int(parts[1].strip())
                    self.utterances.append((utt, label))

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, index):
        utt, label = self.utterances[index]
        x, fs = load_audio(utt)
        x = normalize_and_apply_sad(x, fs, utt)
        x = pad_random(x, Settings().network.train_clip_size)
        if Settings().rawboost.enabled and random.random() < Settings().rawboost.applyingRatio:
            x = rawboost.apply_rawboost(x, fs, Settings().rawboost)
        x = Tensor(x)
        return x, label


class TrialListDataset(Dataset):

    def __init__(self, batches, dataset_folder):
        self.batches = batches
        self.dataset_folder = dataset_folder

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        signal_list = []
        key_list = []
        batch, fixed_length = self.batches[index]
        for key in batch:
            x, fs = load_audio(os.path.join(self.dataset_folder, key))
            x = normalize_and_apply_sad(x, fs, key)
            x = x[:fixed_length]
            x = tile_to_size(x, Settings().network.min_evaluation_utterance_length_in_samples)
            x = Tensor(x)
            signal_list.append(x)
            key_list.append(key)
        signals = torch.stack(signal_list)
        return signals, key_list


class TrialListSadComputationDataset(Dataset):
    def __init__(self, utts: List[str], dataset_folder):
        self.utterances = utts
        self.dataset_folder = dataset_folder

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, index):
        key = self.utterances[index]
        x, fs = load_audio(os.path.join(self.dataset_folder, key))
        x = normalize_and_apply_sad(x, fs, key)
        return x.size

def chunk_into_n(lst, n):
  size = ceil(len(lst) / n)
  return list(
    map(lambda x: lst[x * size:x * size + size],
    list(range(n)))
  )

class TrialList:
    def __init__(self, dataset_folder: str, trial_list_file: str, score_filename: str = None):
        self.dataloader = None
        self.max_n_utts_in_splits = None
        self.n_utts_in_splits = None
        self.utts_in_score_order = None
        self.utts = []
        self.trial_list_file = trial_list_file
        self.dataset_folder = dataset_folder
        self.filename = self.trial_list_file.rsplit('/', maxsplit=1)[-1]
        save_folder = fileutils.get_post_sad_lengths_folder()
        self.sadLengthsFile = os.path.join(save_folder, self.filename) + '.npy'
        if score_filename is None:
            self.score_filename = self.filename
        else:
            self.score_filename = score_filename

        with open(trial_list_file, "r") as f_trl:
            self.trialLines = f_trl.readlines()
        for line in self.trialLines:
            key, _ = line.strip().split(" ")
            self.utts.append(key)

    def compute_post_sad_lengths(self):
        batches = chunk_into_n(self.utts, Settings().computing.world_size)
        utterance_batch = batches[Settings().computing.global_process_rank]
        len_batch = len(batches[0])
        len_last_batch = len(batches[-1])

        dataset = TrialListSadComputationDataset(utterance_batch, self.dataset_folder)
        sad_dataloader = DataLoader(dataset,
                                     batch_size=10,
                                     shuffle=False,
                                     drop_last=False,
                                     pin_memory=False,
                                     num_workers=Settings().computing.network_dataloader_workers)

        if Settings().computing.global_process_rank == 0:
            print("Determining utterance lengths for scoring... [{}]".format(self.trial_list_file))
        local_lengths = torch.zeros(len_batch)
        i = 0
        for j, lengths in enumerate(sad_dataloader):
            local_lengths[i:i + torch.numel(lengths)] = lengths
            i += torch.numel(lengths)
            if Settings().computing.global_process_rank == 0 and j % 500 == 0:
                print(str(j) + '/' + str(len(sad_dataloader)) + " {}".format(i))

        local_lengths = local_lengths.to(Settings().computing.device)
        if Settings().computing.global_process_rank == 0:
            if Settings().computing.world_size > 1:
                print("Gathering lengths...")
                gathered_lengths = [torch.zeros(len_batch, device=Settings().computing.device) for _ in
                                   range(Settings().computing.world_size)]
                gather(local_lengths, gathered_lengths)
            else:
                gathered_lengths = [local_lengths]

            gathered_lengths = [tensor.tolist() for tensor in gathered_lengths]
            gathered_lengths = [x for y in gathered_lengths for x in y]
            if len_batch != len_last_batch:
                gathered_lengths = gathered_lengths[:(len_last_batch - len_batch)]

            gathered_lengths = np.asarray(gathered_lengths, dtype=int)

            np.save(self.sadLengthsFile, gathered_lengths)
            print('File lengths saved to: {}'.format(self.sadLengthsFile))

        else:
            gather(local_lengths)

    def init_data_loader(self):
        print('Initializing dataloader for {}...'.format(self.trial_list_file))
        max_cut_proportion = Settings().network.max_cut_proportion
        min_utt_length = Settings().network.min_evaluation_utterance_length_in_samples
        #max_utt_length = Settings().network.max_evaluation_utterance_length_in_samples
        max_batch_size_in_samples = Settings().network.minibatch_size * Settings().network.eval_minibatch_size_factor * Settings().network.train_clip_size
        max_utt_length = min(Settings().network.max_evaluation_utterance_length_in_samples, max_batch_size_in_samples)
        lengths = np.load(self.sadLengthsFile)
        sorted_indices = np.argsort(lengths)
        lengths = lengths[sorted_indices]
        #print(lengths)
        n_utts = len(self.utts)
        batch_samples_sum = 0
        batch_size_sum = 0
        index = 0
        batch_data = []
        while index < n_utts:
            #print(lengths[index], min_utt_length)
            fixed_segment_length = max(lengths[index], min_utt_length)
            #print('Creating batches of length {}'.format(fixed_segment_length))
            if fixed_segment_length > max_utt_length:
                fixed_segment_length = max_utt_length
            max_segment_length = fixed_segment_length / (1 - max_cut_proportion)
            if max_segment_length >= max_utt_length:
                max_segment_length *= 100
            samples_filled = 0
            batch = []
            while samples_filled + fixed_segment_length <= max_batch_size_in_samples:
                #print('hei')
                samples_filled += fixed_segment_length
                batch.append(self.utts[sorted_indices[index]])
                index += 1
                if index == n_utts or lengths[index] > max_segment_length:
                    break
            batch_samples_sum += samples_filled
            batch_size_sum += len(batch)
            batch_data.append((batch, fixed_segment_length))

        if Settings().computing.global_process_rank == 0:
            print('{} Testing minibatches created!'.format(len(batch_data)))
            print('  - Maximum proportion of cutted speech (setting): {} %'.format(max_cut_proportion * 100))
            print('  - Maximum batch size in samples (setting): {}'.format(max_batch_size_in_samples))
            print('  - Average batch size in samples (realized): {:.1f}'.format(batch_samples_sum / len(batch_data)))
            print('  - Average batch size in utterances (realized): {:.1f}'.format(batch_size_sum / len(batch_data)))

        gpu_splits = [batch_data[i::Settings().computing.world_size] for i in range(Settings().computing.world_size)]
        gpu_split = gpu_splits[Settings().computing.global_process_rank]
        utts_in_gpu_splits = [[utt for batch in gpu_split for utt in batch[0]] for gpu_split in gpu_splits]
        self.utts_in_score_order = [utt for gpu_split in utts_in_gpu_splits for utt in gpu_split]
        self.n_utts_in_splits = [len(gpu_split) for gpu_split in utts_in_gpu_splits]
        self.max_n_utts_in_splits = max(self.n_utts_in_splits)

        dataset = TrialListDataset(gpu_split, self.dataset_folder)
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, collate_fn =_collater,
                                     pin_memory=True,
                                     num_workers=Settings().computing.network_dataloader_workers)
        print('Dataloader initialized!')

    def score(self, model, prefix: str) -> str:
        model.eval()
        local_scores = torch.zeros(self.max_n_utts_in_splits).to(Settings().computing.device)
        i = 0
        if Settings().computing.global_process_rank == 0:
            print("Scoring...")
        for j, (batch_x, utt_id) in enumerate(self.dataloader):
            batch_x = batch_x.to(Settings().computing.device)
            with torch.no_grad():
                _, batch_out = model(batch_x)
                local_scores[i:i+len(utt_id)] = (batch_out[:, 1]).data
            i += len(utt_id)
            if j % Settings().network.scoring_print_interval == 0 and Settings().computing.global_process_rank == 0:
                print('GPU {}: '.format(Settings().computing.global_process_rank) + str(j) + '/' + str(len(self.dataloader)))
        model.train()
        model = None
        torch.cuda.empty_cache()

        if Settings().computing.global_process_rank == 0:
            if Settings().computing.world_size > 1:
                print("Gathering scores...")
                gathered_scores = [torch.zeros(self.max_n_utts_in_splits, device= Settings().computing.device) for _ in range(Settings().computing.world_size)]
                gather(local_scores, gathered_scores)
            else:
                gathered_scores = [local_scores]
        else:
            gather(local_scores)

        output_file = fileutils.get_score_output_file(self.score_filename, prefix)

        if Settings().computing.global_process_rank == 0:
            gathered_scores = [tensor.tolist()[:self.n_utts_in_splits[i]] for i, tensor in enumerate(gathered_scores)]
            gathered_scores = [x for y in gathered_scores for x in y]

            score_dict = {}
            for i, utt in enumerate(self.utts_in_score_order):
                score_dict[utt] = gathered_scores[i]

            print("Writing a score file...")

            with open(output_file, "w") as out:
                for trl in self.trialLines:
                    utt_id, key = trl.strip().split(' ')
                    out.write("{} {} {}\n".format(utt_id, key, score_dict[utt_id]))

        if Settings().computing.world_size > 1:
            torch.distributed.barrier()

        return output_file


def _collater(batch):
    # For trial list batch is already formed in the DataSet object.
    return batch[0]
