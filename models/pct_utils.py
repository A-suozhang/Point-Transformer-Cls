import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2_utils import furthest_point_sample as farthest_point_sample_cuda
from pointnet2_utils import gather_operation as index_points_cuda_transpose
from pointnet2_utils import grouping_operation as grouping_operation_cuda
from pointnet2_utils import ball_query as query_ball_point_cuda

from knn_cuda import KNN

def index_points_cuda(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    points = points.transpose(1,2).contiguous() #[B, C, N]
    new_points = index_points_cuda_transpose(points, idx) #[B, C, S]
    
    return new_points.transpose(1,2).contiguous()


def stem_knn(xyz, points, k):
    knn = KNN(k=k, transpose_mode=True)
    xyz = xyz.permute([0,2,1])
    _, idx = knn(xyz.contiguous(), xyz) # xyz: [bs, npoints, coord] idx: [bs, npoint, k]
    idx = idx.int()

    grouped_xyz = grouping_operation_cuda(xyz.transpose(1,2).contiguous(), idx) # [bs, xyz, n_point, k]
    grouped_points = grouping_operation_cuda(points.contiguous(), idx) #B, C, npoint, k)

    return grouped_xyz, grouped_points


def sample_and_group_cuda(npoint, k, xyz, points):
    """
    Input:
        npoint:
        k:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, C, N]
    Return:
        new_xyz: sampled points position data, [B, 3, npoint]
        new_points: sampled points data, [B, C+C_xyz, npoint, k]
        grouped_xyz_norm: sampled relative points position data, [B, 3, npoint, k]
    """
    knn = KNN(k=k, transpose_mode=True)

    B, N, C_xyz = xyz.shape

    if npoint < N:
        fps_idx = farthest_point_sample_cuda(xyz, npoint) # [B, npoint]
        torch.cuda.empty_cache()
        new_xyz = index_points_cuda(xyz, fps_idx) #[B, npoint, 3]
    else:
        new_xyz = xyz

    
    torch.cuda.empty_cache()
    _, idx = knn(xyz.contiguous(), new_xyz) # B, npoint, k
    idx = idx.int()
    
    torch.cuda.empty_cache()
    grouped_xyz = grouping_operation_cuda(xyz.transpose(1,2).contiguous(), idx).permute(0,2,3,1) # [B, npoint, k, C_xyz]
    #print(grouped_xyz.size())
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, npoint, 1, C_xyz) # [B, npoint, k, 3]
    grouped_xyz_norm = grouped_xyz_norm.permute(0,3,1,2).contiguous()# [B, 3, npoint, k]
    torch.cuda.empty_cache()

    grouped_points = grouping_operation_cuda(points.contiguous(), idx) #B, C, npoint, k

    new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=1) # [B, C+C_xyz, npoint, k]
    

    return new_xyz.transpose(1,2), grouped_xyz_norm, new_points

class TDLayer(nn.Module):
    def __init__(self, npoint, input_dim, out_dim, k=16):
        super().__init__()
        '''
        Transition Down Layer
        npoint: number of input points
        nsample: k in kNN, default 16
        in_dim: feature dimension of the input feature x (output of the PCTLayer)
        out_dim: feature dimension of the TDLayer

        '''
        self.npoint = npoint
        self.k = k
        self.input_dim = input_dim
        self.out_dim = out_dim

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        self.mlp_convs.append(nn.Conv2d(input_dim+3, input_dim, 1))
        self.mlp_convs.append(nn.Conv2d(input_dim, out_dim, 1))
        self.mlp_bns.append(nn.BatchNorm2d(input_dim))
        self.mlp_bns.append(nn.BatchNorm2d(out_dim))

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, 3, N]
            points: input points data, [B, C, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B, input_dim, npoint = list(xyz.size())
        xyz = xyz.permute(0, 2, 1)


        new_xyz, grouped_xyz_norm, new_points = sample_and_group_cuda(self.npoint, self.k, xyz, points)
        # new_xyz: sampled points position data, [B, 3, npoint]
        # new_points: sampled points data, [B, C+C_xyz, npoint,k]
        # grouped_xyz_norm: [B, 3, npoint,k]

        #new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        #print(new_points.size())
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))


        new_points_pooled = torch.max(new_points, 3)[0] # local max pooling
        #new_xyz = new_xyz.permute(0, 2, 1)
        #print(new_points_pooled.size())
        return new_xyz, new_points_pooled, grouped_xyz_norm, new_points

# class PTBlock(nn.Module):
    # def __init__(self, in_dim, is_firstlayer=False):
        # super().__init__()
        # '''
        # Point Transformer Layer

        # in_dim: feature dimension of the input feature x
        # out_dim: feature dimension of the Point Transformer Layer
        # '''


        # self.in_dim = in_dim
        # self.is_firstlayer = is_firstlayer
        # # self.hidden_dim = int(in_dim/2)
        # self.hidden_dim = in_dim

        # self.linear_top = nn.Sequential(
            # nn.Conv1d(in_dim, self.hidden_dim, 1), 
            # nn.BatchNorm1d(self.hidden_dim))
        # self.linear_down = nn.Sequential(
            # nn.Conv1d(self.hidden_dim, self.in_dim, 1), 
            # nn.BatchNorm1d(self.in_dim))

        # self.phi = nn.Sequential(
            # nn.Conv1d(self.hidden_dim, self.hidden_dim, 1), 
            # nn.BatchNorm1d(self.hidden_dim))
        # self.psi = nn.Sequential(
            # nn.Conv2d(self.in_dim, self.hidden_dim, 1),
            # nn.BatchNorm2d(self.hidden_dim))
        # self.alpha = nn.Sequential(
            # nn.Conv2d(self.in_dim, self.hidden_dim, 1), 
            # nn.BatchNorm2d(self.hidden_dim))

        # self.gamma = nn.Sequential(
            # nn.Conv2d(self.hidden_dim, self.hidden_dim, 1), 
            # nn.BatchNorm2d(self.hidden_dim), 
            # nn.ReLU(), 
            # nn.Conv2d(self.hidden_dim, self.hidden_dim, 1), 
            # nn.BatchNorm2d(self.hidden_dim))
        # self.delta = nn.Sequential(
            # nn.Conv2d(3, self.hidden_dim, 1), 
            # nn.BatchNorm2d(self.hidden_dim), 
            # nn.ReLU(), 
            # nn.Conv2d(self.hidden_dim, self.hidden_dim, 1), 
            # nn.BatchNorm2d(self.hidden_dim)
            # )

    # def forward(self, input_p_centroids, input_x_centroids, input_p, input_x):
        # '''
        # input_x: B, in_dim, npoint, nsample
        # input_p: B, 3, npoint, nsample
        # input_x_centroids: B, in_dim, npoint
        # input_p_centroids: B, 3, npoint
        # '''
        # B, in_dim, npoint, nsample = list(input_x.size())
        # tmp = input_x_centroids

        # input_x_centroids = self.linear_top(input_x_centroids)

        # #print(input_p_centroids.size(), input_x_centroids.size(), self.phi)

        # phi_xi = self.phi(input_x_centroids) # B, hidden_dim, npoint
        # phi_xi = phi_xi.view(B, self.hidden_dim, npoint, 1).repeat(1,1,1,nsample) # B, hidden_dim, npoint, nsample

        # # input_x - [bs, feature, npoints, l]
        # # psi - [bs, o_feature, npoints, l]
        # psi_xj = self.psi(input_x) # B, hidden_dim, npoint, nsample
        # alpha_xj = self.alpha(input_x) # B, hidden_dim, npoint, nsample

        # pipj = input_p_centroids.view(B,3,npoint,1).repeat(1,1,1,nsample) - input_p  # B, 3, npoint, nsample
        # delta_p = self.delta(pipj) # B, hidden_dim, npoint, nsample

        # y = F.softmax(self.gamma(phi_xi - psi_xj + delta_p), dim=-1)*(alpha_xj + delta_p) # B, hidden_dim, npoint, nsample
        # y = y.sum(dim=-1)# B, hidden_dim, npoint
        # y = self.linear_down(y)# B, input_dim, npoint

        # return y+tmp

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)



class PTBlock(nn.Module):
    def __init__(self, in_dim, is_firstlayer=False, n_sample=16):
        super().__init__()
        '''
        Point Transformer Layer

        in_dim: feature dimension of the input feature x
        out_dim: feature dimension of the Point Transformer Layer(currently same with hidden-dim)
        [?] - not sure how to set hidden. the paper only gives the out
        '''


        self.in_dim = in_dim
        self.is_firstlayer = is_firstlayer
        # ??: what is the hidden_dim
        # self.hidden_dim = int(in_dim/2)
        self.hidden_dim = in_dim
        self.n_sample = n_sample

        self.linear_top = nn.Sequential(
            nn.Conv1d(in_dim, self.hidden_dim, 1), 
            nn.BatchNorm1d(self.hidden_dim))
        self.linear_down = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.in_dim, 1), 
            nn.BatchNorm1d(self.in_dim))

        self.phi = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, 1), 
            nn.BatchNorm1d(self.hidden_dim))
        self.psi = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, 1),
            nn.BatchNorm1d(self.hidden_dim))
        self.alpha = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, 1), 
            nn.BatchNorm1d(self.hidden_dim))

        self.gamma = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 1), 
            nn.BatchNorm2d(self.hidden_dim), 
            nn.ReLU(), 
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 1), 
            nn.BatchNorm2d(self.hidden_dim))
        self.delta = nn.Sequential(
            nn.Conv2d(3, self.hidden_dim, 1), 
            nn.BatchNorm2d(self.hidden_dim), 
            nn.ReLU(), 
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 1), 
            nn.BatchNorm2d(self.hidden_dim)
            )

        self.knn = KNN(k=n_sample, transpose_mode=True)

    def forward(self, input_p, input_x, a, b):
        '''
        input_p:  B, 3, npoint
        input_x: B, in_dim, npoint
        '''

        B, in_dim, npoint = list(input_x.size())
        n_sample = self.n_sample

        res = input_x

        input_p = input_p.permute([0,2,1])

        # DEBUG: error here is that in the last block only 4-points;
        # however the knn still gives 16 idxs
        if npoint < self.n_sample:
            self.knn = KNN(k=npoint, transpose_mode=True)

        _, idx = self.knn(input_p.contiguous(), input_p)
        _, idx = self.knn(input_p, input_p)
        idx = idx.int()

        grouped_input_p = grouping_operation_cuda(input_p.transpose(1,2).contiguous(), idx) # [bs, xyz, npoint, k]

        input_x = self.linear_top(input_x)
        # grouped_input_x = index_points(input_x.permute([0,2,1]), idx.long()).permute([0,3,1,2])
        # grouped_input_x = grouping_operation_cuda(input_x.contiguous(), idx)  # [bs, xyz, npoint, K]
        phi = self.phi(input_x)
        phi = phi[:,:,:,None].repeat(1,1,1,min(self.n_sample, npoint))
        psi = grouping_operation_cuda(self.psi(input_x).contiguous(), idx)
        alpha = grouping_operation_cuda(self.alpha(input_x).contiguous(), idx) # [bs, xyz, npoint, k]

        relative_xyz = input_p.permute([0,2,1])[:,:,:,None] - grouped_input_p
        pos_encoding = self.delta(relative_xyz)    # [bs, dims, npoint, k]

        y = F.softmax(self.gamma(phi - psi + pos_encoding), dim=-1)*(alpha + pos_encoding)
        y = y.sum(dim=-1)

        y = self.linear_down(y)

        return y+res

