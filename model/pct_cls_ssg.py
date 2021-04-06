import torch
import torch.nn as nn
import torch.nn.functional as F
from pct_utils import TDLayer, TULayer, PTBlock, stem_knn


class get_model(nn.Module):
    def __init__(self,num_class,N,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.input_mlp = nn.Sequential(
            nn.Conv1d(in_channel, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, 1),
            nn.BatchNorm1d(32))

        self.in_dims = [32, 64, 128, 256]
        self.out_dims = [64, 128, 256, 512]
        self.neighbor_ks = [16, 32, 64, 16, 16]

        self.PTBlock0 = PTBlock(in_dim=self.in_dims[0], n_sample=self.neighbor_ks[0])

        self.TDLayer1 = TDLayer(npoint=int(N/4),input_dim=self.in_dims[0], out_dim=self.out_dims[0], k=self.neighbor_ks[1])
        self.PTBlock1 = PTBlock(in_dim=self.out_dims[0], n_sample=self.neighbor_ks[1])

        self.TDLayer2 = TDLayer(npoint=int(N/16),input_dim=self.in_dims[1], out_dim=self.out_dims[1], k=self.neighbor_ks[2])
        self.PTBlock2 = PTBlock(in_dim=self.out_dims[1], n_sample=self.neighbor_ks[2])

        self.TDLayer3 = TDLayer(npoint=int(N/64),input_dim=self.in_dims[2], out_dim=self.out_dims[2], k=self.neighbor_ks[3])
        self.PTBlock3 = PTBlock(in_dim=self.out_dims[2], n_sample=self.neighbor_ks[3])

        self.TDLayer4 = TDLayer(npoint=int(N/256),input_dim=self.in_dims[3], out_dim=self.out_dims[3], k=self.neighbor_ks[4])
        # self.TDLayer4 = TDLayer(npoint=int(N/64),input_dim=256, out_dim=512, k=self.neighbor_ks[4])
        self.PTBlock4 = PTBlock(in_dim=self.out_dims[3], n_sample=self.neighbor_ks[4])

        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, num_class)

        self.use_ln = False
        if self.use_ln:
            self.final_ln = nn.LayerNorm(256)

        self.save_flag = False
        self.save_dict = {}
        for i in range(5):
            self.save_dict['attn_{}'.format(i)] = []

    def save_intermediate(self):

        save_dict = self.save_dict
        self.save_dict = {}
        for i in range(5):
            self.save_dict['attn_{}'.format(i)] = []
        return save_dict


    def forward(self, inputs):
        B,_,_ = list(inputs.size())

        if self.normal_channel:
            l0_xyz = inputs[:, :3, :]
        else:
            l0_xyz = inputs

        input_points = self.input_mlp(inputs)
        l0_points, attn_0 = self.PTBlock0(l0_xyz, input_points)

        l1_xyz, l1_points, l1_xyz_local, l1_points_local = self.TDLayer1(l0_xyz, l0_points)
        l1_points, attn_1 = self.PTBlock1(l1_xyz, l1_points)

        l2_xyz, l2_points, l2_xyz_local, l2_points_local = self.TDLayer2(l1_xyz, l1_points)
        l2_points, attn_2 = self.PTBlock2(l2_xyz, l2_points)

        l3_xyz, l3_points, l3_xyz_local, l3_points_local = self.TDLayer3(l2_xyz, l2_points)
        l3_points, attn_3 = self.PTBlock3(l3_xyz, l3_points)

        l4_xyz, l4_points, l4_xyz_local, l4_points_local = self.TDLayer4(l3_xyz, l3_points)
        l4_points, attn_4 = self.PTBlock4(l4_xyz, l4_points)

        if self.save_flag:
            self.save_dict['attn_0'].append(attn_0)
            self.save_dict['attn_1'].append(attn_1)
            self.save_dict['attn_2'].append(attn_2)
            self.save_dict['attn_3'].append(attn_3)
            self.save_dict['attn_4'].append(attn_4)

        del attn_0
        del attn_1
        del attn_2
        del attn_3
        del attn_4

        l4_points = l4_points.mean(dim=-1)

        x = l4_points.view(B, -1)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        # apply the final LN for pre-LN scheme
        if self.use_ln:
            x = self.final_ln(x)

        x = self.fc2(x)
        x = F.log_softmax(x, -1)

        return x


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, smooth=True):
        # dont know why the net output adds a torch.log_softmax after the logits, and here uses nll_loss
        if smooth:
            eps = 0.2
            n_class = pred.shape[1]
            one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            total_loss = -(one_hot * pred).sum(dim=1).mean()
        else:
            total_loss = F.nll_loss(pred, target)

        return total_loss
