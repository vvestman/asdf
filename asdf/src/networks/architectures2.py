# Copyright 2020 Ville Vestman
#           2020 Kong Aik Lee
#           2022 Hemlata Tak
#           2024 Ville Vestman
# This file is licensed under the MIT license (see LICENSE.txt).
import os

import torch
import torch.nn as nn
import fairseq
import torch.nn.functional as F


from asdf.src.misc import fileutils
from asdf.src.networks.modules import *
from asdf.src.networks.modules2 import *
from asdf.src.settings.settings import Settings

class BaseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.training_loss = torch.nn.Parameter(torch.Tensor([torch.finfo().max]), requires_grad=False)
        self.consecutive_lr_updates = torch.nn.Parameter(torch.LongTensor([0]), requires_grad=False)



class SSLModel(nn.Module):
    def __init__(self, device):
        super(SSLModel, self).__init__()

        # TODO download model
        cp_path = os.path.join(fileutils.get_folder_of_file(__file__), 'xlsr2_300m.pt')  # Change the pre-trained XLSR model path.
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device = device
        self.out_dim = 1024
        return

    def extract_feat(self, input_data):

        # put the model to GPU if it not there
        if next(self.model.parameters()).device != input_data.device \
                or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()

        if True:
            # input should be in shape (batch, length)
            if input_data.ndim == 3:
                input_tmp = input_data[:, :, 0]
            else:
                input_tmp = input_data

            # [batch, length, dim]
            emb = self.model(input_tmp, mask=False, features_only=True)['x']
        return emb


class DefaultNetwork(BaseNet):
    def __init__(self):
        super().__init__()
        self.device = Settings().computing.device

        self.ssl_model = SSLModel(self.device)
        self.LL = nn.Linear(self.ssl_model.out_dim, 128)



        self.bn1 = nn.BatchNorm1d(num_features=128)

        self.cnn1 = CnnLayer2d(128, 128, 3, 1)
        self.seLayer1 = SELayer(128)
        self.cnn2 = CnnLayer2d(128, 128, 5, 1)
        self.seLayer2 = SELayer(128)


        self.pooling1 = MeanMaxMinStdPoolingLayer()

        self.LL2 = nn.Linear(128 * 4, 128)
        self.bn2 = nn.BatchNorm1d(num_features=128)

        #self.pooling2 = MeanMaxMinStdPoolingLayer()

        #self.cnn3 = CnnLayer(128, 128, 5, 1)
        self.LL3 = nn.Linear(128, 128)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.LL4 = nn.Linear(128, 2)

        self.LLw = nn.Linear(self.ssl_model.out_dim, 128)
        self.bnw1 = nn.BatchNorm1d(num_features=128)
        self.cnnw1 = CnnLayer2d(128, 1, 5, 1)










    def forward(self, x):
        # -------pre-trained Wav2vec model fine tunning ------------------------##
        with torch.no_grad():
            x = self.ssl_model.extract_feat(x.squeeze(-1))
        #print(x.size())

        xw = self.LLw(x)
        xw = xw.transpose(1, 2)  # (bs,feat_out_dim,frame_number)
        xw = self.bnw1(xw)
        xw = F.selu(xw)
        xw = xw.unfold(2, 20, 2)
        xw = self.cnnw1(xw)
        xw = torch.mean(xw, dim=3)
        xw = torch.squeeze(xw, dim=1)
        #print(xw.size())
        xw = torch.softmax(xw, dim=1)
        #print(xw)


        x = self.LL(x)  # (bs,frame_number,feat_out_dim)
        x = x.transpose(1, 2)  # (bs,feat_out_dim,frame_number)
        x = self.bn1(x)
        x = F.selu(x)
        #print(x.size())
        # post-processing on front-end features
        x = x.unfold(2, 20, 2)
        #print(x.size())
        x = self.cnn1(x)
        x = self.seLayer1(x)
        #print(x.size())
        x = self.cnn2(x)
        x = self.seLayer2(x)

        #print(x.size())
        x = self.pooling1(x)
        #print(x.size())
        x = x.transpose(1, 2)
        x = self.LL2(x)  # (bs,frame_number,feat_out_dim)
        x = x.transpose(1, 2)  # (bs,feat_out_dim,frame_number)
        x = self.bn2(x)
        x = F.selu(x)
        #print(x.size())
        #x = self.pooling1(x)
        #print(x.size(), xw.size())
        x = torch.bmm(x, xw.unsqueeze(2)).squeeze(2)
        #x = torch.mean(x, dim=2)
        #print(x.size())
        x = self.LL3(x)
        x = self.bn3(x)
        x = F.selu(x)
        #print(x.size())
        x = self.LL4(x)

        return x, x
        #return last_hidden, output




