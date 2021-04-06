import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2_utils import furthest_point_sample as farthest_point_sample_cuda
from pointnet2_utils import gather_operation as index_points_cuda_transpose
from pointnet2_utils import grouping_operation as grouping_operation_cuda
from pointnet2_utils import ball_query as query_ball_point_cuda

from knn_cuda import KNN

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


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
    
    # take in [B, 3, N]
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
    k = min(npoint, k)
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

class TULayer(nn.Module):
    def __init__(self, npoint, input_dim, out_dim, k=3):
        super().__init__()
        '''
        Transition Up Layer
        npoint: number of input points
        nsample: k in kNN, default 3
        in_dim: feature dimension of the input feature x (output of the PCTLayer)
        out_dim: feature dimension of the TDLayer

        '''
        self.npoint = npoint
        self.k = k
        self.input_dim = input_dim
        self.out_dim = out_dim

        self.linear_1 = nn.Conv1d(input_dim, out_dim, 1)
        self.linear_2 = nn.Conv1d(out_dim, out_dim, 1)

    def forward(self, xyz_1, xyz_2, points_1, points_2):
        """
        Input:
            M < N
            xyz_1: input points position data, [B, 3, M]
            xyz_2: input points position data, [B, 3, N]
            points_1: input points data, [B, C, M]
            points_2: input points data, [B, C, N]

            interpolate xyz_2's coordinates feature with knn neighbor's features weighted by inverse distance

        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """

        B, input_dim, M = list(points_1.size())
        B, output_dim, N = list(points_2.size())

        points_1 = self.linear_1(points_1)
        points_2 = self.linear_2(points_2)


        dists = square_distance(xyz_2.transpose(1,2), xyz_1.transpose(1,2)) # [B, N, M]
        dists, idx = dists.sort(dim=-1)
        dists, idx = dists[:,:,:self.k], idx[:,:,:self.k]

        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_points = torch.sum( \
                        grouping_operation_cuda(points_1, idx.int())*weight.view(B, 1, N, 3)
                                                ,dim=-1)


        return xyz_2 , (interpolated_points + points_2)

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

class TransposeLayerNorm(nn.Module):

    def __init__(self, dim):
        super(TransposeLayerNorm, self).__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        if len(x.shape) == 3:
            # [bs, in_dim, npoints]
            pass
        elif len(x.shape) == 4:
            # [bs, in_dim, npoints, k]
            pass
        else:
            raise NotImplementedError

        return self.norm(x.transpose(1,-1)).transpose(1,-1)

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

        # TODO: set the hidden/vector/out_dims
        self.hidden_dim = in_dim
        # self.out_dim = min(4*in_dim, 512)
        self.out_dim = in_dim
        self.vector_dim = self.out_dim
        self.n_sample = n_sample

        # whether use BN or LN or None
        # 0 - None
        # 1 - BN
        # 2 - LN

        self.use_bn = 1
        # use transformer-like preLN before the attn & ff layer
        self.pre_ln = False

        # whether to use the vector att or the original attention
        self.use_vector_attn = True
        self.nhead = 4

        self.linear_top = nn.Sequential(
            nn.Conv1d(in_dim, self.hidden_dim, 1),
            # TransposeLayerNorm(self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim) if self.use_bn else nn.Identity()
        )
        self.linear_down = nn.Sequential(
            nn.Conv1d(self.out_dim, self.in_dim, 1),
            # TransposeLayerNorm(self.in_dim),
            nn.BatchNorm1d(self.in_dim) if self.use_bn else nn.Identity()
        )

        self.phi = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.out_dim, 1),
            # nn.BatchNorm1d(self.out_dim) if self.use_bn else nn.Identity()
        )
        self.psi = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.out_dim, 1),
            # nn.BatchNorm1d(self.out_dim) if self.use_bn else nn.Identity()
        )
        self.alpha = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.out_dim, 1),
            # nn.BatchNorm1d(self.out_dim) if self.use_bn else nn.Identity()
        )

        self.gamma = nn.Sequential(
            nn.Conv2d(self.out_dim, self.hidden_dim, 1),
            # TransposeLayerNorm(self.hidden_dim),
            nn.BatchNorm2d(self.hidden_dim) if self.use_bn else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, self.vector_dim, 1),
            # TransposeLayerNorm(self.vector_dim),
            nn.BatchNorm2d(self.vector_dim) if self.use_bn else nn.Identity()
        )

        self.delta = nn.Sequential(
            nn.Conv2d(3, self.hidden_dim, 1),
            # TransposeLayerNorm(self.hidden_dim),
            nn.BatchNorm2d(self.hidden_dim) if self.use_bn else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, self.out_dim, 1),
            nn.BatchNorm2d(self.out_dim) if self.use_bn else nn.Identity()
            # TransposeLayerNorm(self.out_dim),
            )

        if self.pre_ln:
            self.ln_top = nn.LayerNorm(self.in_dim)
            self.ln_attn = nn.LayerNorm(self.hidden_dim)
            self.ln_down = nn.LayerNorm(self.out_dim)

        self.knn = KNN(k=n_sample, transpose_mode=True)

    def forward(self, input_p, input_x):
        '''
        input_p:  B, 3, npoint
        input_x: B, in_dim, npoint
        '''

        B, in_dim, npoint = list(input_x.size())
        n_sample = self.n_sample
        k = min(n_sample, npoint)
        h = self.nhead

        res = input_x

        input_p = input_p.permute([0,2,1])

        # DEBUG: error here is that in the last block only 4-points;
        # however the knn still gives 16 idxs
        if npoint < self.n_sample:
            self.knn = KNN(k=npoint, transpose_mode=True)

        # DEBUG ONLY: using the input_x: feature space knn!
        # _, idx = self.knn(input_x.transpose(1,2), input_x.transpose(1,2))

        _, idx = self.knn(input_p.contiguous(), input_p)
        idx = idx.int()

        grouped_input_p = grouping_operation_cuda(input_p.transpose(1,2).contiguous(), idx) # [bs, xyz, npoint, k]

        if self.pre_ln:
            input_x = self.ln_top(input_x.transpose(1,2)).transpose(1,2)

        input_x = self.linear_top(input_x)

        # TODO: apply the layer-norm
        # however the original is [bs, dim, npoint]
        if self.pre_ln:
            input_x = self.ln_attn(input_x.transpose(1,2)).transpose(1,2)

        # grouped_input_x = index_points(input_x.permute([0,2,1]), idx.long()).permute([0,3,1,2])
        # grouped_input_x = grouping_operation_cuda(input_x.contiguous(), idx)  # [bs, xyz, npoint, K]
        phi = self.phi(input_x)
        phi = phi[:,:,:,None].repeat(1,1,1,k)
        psi = grouping_operation_cuda(self.psi(input_x).contiguous(), idx)
        alpha = grouping_operation_cuda(self.alpha(input_x).contiguous(), idx) # [bs, xyz, npoint, k]

        relative_xyz = input_p.permute([0,2,1])[:,:,:,None] - grouped_input_p
        pos_encoding = self.delta(relative_xyz)    # [bs, dims, npoint, k]

        if self.use_vector_attn:
            # the attn_map: [vector_dim];
            # the alpha:    [out_dim]
            attn_map = F.softmax(self.gamma(phi - psi + pos_encoding), dim=-1)
            y = attn_map.repeat(1, self.out_dim // self.vector_dim,1,1)*(alpha + pos_encoding)
            y = y.sum(dim=-1)
        else:
            phi = phi.reshape(B, h, self.out_dim//h, npoint, k)
            psi = psi.reshape(B, h, self.out_dim//h, npoint, k)
            attn_map = F.softmax((phi*psi).reshape(B, self.out_dim, npoint, k) + pos_encoding, dim=-1)
            y = attn_map*(alpha+pos_encoding)
            y = y.sum(dim=-1)

        if self.pre_ln:
            y = self.ln_down(y.transpose(1,2)).transpose(1,2)

        y = self.linear_down(y)

        return y+res, attn_map.detach().cpu().data

