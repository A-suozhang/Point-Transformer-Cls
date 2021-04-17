# Copyright (c) 2020 NVIDIA CORPORATION.
# Copyright (c) 2018-2020 Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import os
import random
import numpy as np
import glob

try:
    import h5py
except:
    print("Install h5py with `pip install h5py`")
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import MinkowskiEngine as ME

import sys
sys.path.append('/home/zhaotianchen/project/point-transformer/pt-cls/model')
from pct_voxel_utils import TDLayer, TULayer, PTBlock

class MinkowskiTransformer(ME.MinkowskiNetwork):
    def __init__(self, in_channel, out_channel, num_class, embedding_channel=1024, dimension=3):
        ME.MinkowskiNetwork.__init__(self, dimension)
        # The normal channel for Modelnet is 3, for scannet is 6, for scanobjnn is 0
        normal_channel = 3
        # in_channel = normal_channel+3 # normal ch + xyz
        self.normal_channel = normal_channel
        self.input_mlp = nn.Sequential(
            ME.MinkowskiConvolution(in_channel, 32, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(32, 32, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(32)
        )

        self.in_dims = [32, 64, 128, 256]
        self.out_dims = [64, 128, 256, 512]
        self.neighbor_ks = [16, 32, 64, 16, 16]
        # self.neighbor_ks = [8, 8, 8, 8, 8]

        self.PTBlock0 = PTBlock(in_dim=self.in_dims[0], n_sample=self.neighbor_ks[0])

        self.TDLayer1 = TDLayer(input_dim=self.in_dims[0], out_dim=self.out_dims[0])
        self.PTBlock1 = PTBlock(in_dim=self.out_dims[0], n_sample=self.neighbor_ks[1])

        self.TDLayer2 = TDLayer(input_dim=self.in_dims[1], out_dim=self.out_dims[1])
        self.PTBlock2 = PTBlock(in_dim=self.out_dims[1], n_sample=self.neighbor_ks[1])

        self.TDLayer3 = TDLayer(input_dim=self.in_dims[2], out_dim=self.out_dims[2])
        self.PTBlock3 = PTBlock(in_dim=self.out_dims[2], n_sample=self.neighbor_ks[2])

        self.TDLayer4 = TDLayer(input_dim=self.in_dims[3], out_dim=self.out_dims[3])
        self.PTBlock4 = PTBlock(in_dim=self.out_dims[3], n_sample=self.neighbor_ks[4])

        self.middle_linear = ME.MinkowskiConvolution(self.out_dims[3], self.out_dims[3], kernel_size=1, dimension=3)
        self.PTBlock_middle = PTBlock(in_dim=self.out_dims[3], n_sample=self.neighbor_ks[4])

        self.TULayer5 = TULayer(input_dim=self.out_dims[3], out_dim=self.in_dims[3])
        self.PTBlock5 = PTBlock(in_dim=self.in_dims[3], n_sample=self.neighbor_ks[4])

        self.TULayer6 = TULayer(input_dim=self.out_dims[2], out_dim=self.in_dims[2])
        self.PTBlock6 = PTBlock(in_dim=self.in_dims[2], n_sample=self.neighbor_ks[3])

        self.TULayer7 = TULayer(input_dim=self.out_dims[1], out_dim=self.in_dims[1])
        self.PTBlock7 = PTBlock(in_dim=self.in_dims[1], n_sample=self.neighbor_ks[2])

        self.TULayer8 = TULayer(input_dim=self.out_dims[0], out_dim=self.in_dims[0])
        self.PTBlock8 = PTBlock(in_dim=self.in_dims[0], n_sample=self.neighbor_ks[1])

        self.fc = nn.Sequential(
            # ME.MinkowskiLinear(32, 32),
            ME.MinkowskiLinear(self.out_dims[3], 32),
            ME.MinkowskiDropout(0.4),
            ME.MinkowskiLinear(32, num_class)
        )

        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()

    def forward(self, in_field: ME.TensorField):

        import time
        start = time.perf_counter()

        x = in_field.sparse()
        x = self.input_mlp(x)

        print('total {} voxels'.format(x.shape[0]))
        x, attn_0 = self.PTBlock0(x)

        x = self.TDLayer1(x)
        x, attn_1 = self.PTBlock1(x)

        x = self.TDLayer2(x)
        x, attn_2 = self.PTBlock2(x)

        x = self.TDLayer3(x)
        x, attn_3 = self.PTBlock3(x)

        x = self.TDLayer4(x)
        # x, attn_4 = self.PTBlock4(x) // at this point it can't find any neighbor with r=10

#        x = self.middle_linear(x)
#        s, attn_middle = self.PTBlock_middle(x)
#
#        x = self.TULayer5(x)
#        x, attn_5 = self.PTBlock5(x)
#
#        x = self.TULayer6(x)
#        x, attn_6 = self.PTBlock6(x)
#
#        x = self.TULayer7(x)
#        x, attn_7 = self.PTBlock7(x)
#
#        x = self.TULayer8(x)
#        x, attn8 = self.PTBlock8(x)

        x = self.global_avg_pool(x)
        x = self.fc(x)

#        out_field = x.slice(in_field)

        end = time.time()

        #print(f"forward time: {end-start} s")
        # print('PT ratio:{}'.format((pt2 - pt1) / (pt2 - pt0)))

        return x




