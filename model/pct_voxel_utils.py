import itertools
import operator
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import numpy.ma as ma

#from pointnet2_utils import furthest_point_sample as farthest_point_sample_cuda
from pointnet2_utils import gather_operation as index_points_cuda_transpose
from pointnet2_utils import grouping_operation as grouping_operation_cuda
from pointnet2_utils import ball_query as query_ball_point_cuda
from pointnet2_utils import QueryAndGroup


#from knn_cuda import KNN
import MinkowskiEngine as ME

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
    def __init__(self, input_dim, out_dim, k=16):
        super().__init__()
        '''
        Transition Down Layer
        npoint: number of input points
        nsample: k in kNN, default 16
        in_dim: feature dimension of the input feature x (output of the PCTLayer)
        out_dim: feature dimension of the TDLayer

        '''
        self.k = k
        self.input_dim = input_dim
        self.out_dim = out_dim

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        self.mlp_convs.append(nn.Conv2d(input_dim+3, input_dim, 1))
        self.mlp_convs.append(nn.Conv2d(input_dim, out_dim, 1))
        self.mlp_bns.append(nn.BatchNorm2d(input_dim))
        self.mlp_bns.append(nn.BatchNorm2d(out_dim))

        self.conv = ME.MinkowskiConvolution(
            input_dim,
            out_dim,
            kernel_size=3,
            stride=2,
            bias=False,
            dimension=3
        )

    def forward(self, x : ME.SparseTensor):
        """
        Input:
            xyz: input points position data, [B, 3, N]
            points: input points data, [B, C, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        # B, input_dim, npoint = list(xyz.size())
        #xyz = xyz.permute(0, 2, 1)

        # new_xyz, grouped_xyz_norm, new_points = sample_and_group_cuda(self.npoint, self.k, xyz, points)
        # new_xyz: sampled points position data, [B, 3, npoint]
        # new_points: sampled points data, [B, C+C_xyz, npoint,k]
        # grouped_xyz_norm: [B, 3, npoint,k]

        # for i, conv in enumerate(self.mlp_convs):
        #     bn = self.mlp_bns[i]
        #     new_points =  F.relu(bn(conv(new_points)))


        # new_points_pooled = torch.max(new_points, 3)[0] # local max pooling
        x = self.conv(x)
        return x

class TULayer(nn.Module):
    def __init__(self, input_dim, out_dim, k=3):
        super().__init__()
        '''
        Transition Up Layer
        npoint: number of input points
        nsample: k in kNN, default 3
        in_dim: feature dimension of the input feature x (output of the PCTLayer)
        out_dim: feature dimension of the TDLayer

        '''
        self.k = k
        self.input_dim = input_dim
        self.out_dim = out_dim

        self.linear_1 = nn.Conv1d(input_dim, out_dim, 1)
        self.linear_2 = nn.Conv1d(out_dim, out_dim, 1)

        self.conv = ME.MinkowskiConvolutionTranspose(
            in_channels=input_dim,
            out_channels=out_dim,
            kernel_size=3,
            stride=2,
            dimension=3
        )

    def forward(self, x : ME.SparseTensor):
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

        # B, input_dim, M = list(points_1.size())
        # B, output_dim, N = list(points_2.size())

        # points_1 = self.linear_1(points_1)
        # points_2 = self.linear_2(points_2)


        # dists = square_distance(xyz_2.transpose(1,2), xyz_1.transpose(1,2)) # [B, N, M]
        # dists, idx = dists.sort(dim=-1)
        # dists, idx = dists[:,:,:self.k], idx[:,:,:self.k]

        # dist_recip = 1.0 / (dists + 1e-8)
        # norm = torch.sum(dist_recip, dim=2, keepdim=True)
        # weight = dist_recip / norm
        # interpolated_points = torch.sum( \
        #                grouping_operation_cuda(points_1, idx.int())*weight.view(B, 1, N, 3)
        #                                       ,dim=-1)


        # return xyz_2 , (interpolated_points + points_2)
        x = self.conv(x)
        return x

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
    def __init__(self, in_dim, is_firstlayer=False, n_sample=16, r=10):
        super().__init__()
        '''
        Point Transformer Layer

        in_dim: feature dimension of the input feature x
        out_dim: feature dimension of the Point Transformer Layer(currently same with hidden-dim)
        [?] - not sure how to set hidden. the paper only gives the out
        '''

        self.r = r # neighborhood cube radius
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
            ME.MinkowskiConvolution(in_dim, self.hidden_dim, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(self.hidden_dim) if self.use_bn else nn.Identity()
        )
        self.linear_down = nn.Sequential(
            ME.MinkowskiConvolution(self.out_dim, self.in_dim, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(self.in_dim) if self.use_bn else nn.Identity()
        )
        # feature transformations
        self.phi = nn.Sequential(
            ME.MinkowskiConvolution(self.hidden_dim, self.out_dim, kernel_size=1, dimension=3)
        )
        self.psi = nn.Sequential(
            ME.MinkowskiConvolution(self.hidden_dim, self.out_dim, kernel_size=1, dimension=3)
        )
        self.alpha = nn.Sequential(
            ME.MinkowskiConvolution(self.hidden_dim, self.out_dim, kernel_size=1, dimension=3)
        )

#        self.gamma = nn.Sequential(
#            ME.MinkowskiConvolution(self.out_dim, self.hidden_dim, kernel_size=1, dimension=3),
#            ME.MinkowskiBatchNorm(self.hidden_dim) if self.use_bn else nn.Identity(),
#            ME.MinkowskiReLU(),
#            ME.MinkowskiConvolution(self.hidden_dim, self.vector_dim, kernel_size=1, dimension=3),
#            ME.MinkowskiBatchNorm(self.vector_dim) if self.use_bn else nn.Identity()
#        )
#
#        self.delta = nn.Sequential(
#            ME.MinkowskiConvolution(3, self.hidden_dim, kernel_size=1, dimension=3),
#            ME.MinkowskiBatchNorm(self.hidden_dim) if self.use_bn else nn.Identity(),
#            ME.MinkowskiReLU(),
#            ME.MinkowskiConvolution(self.hidden_dim, self.out_dim, kernel_size=1, dimension=3),
#            ME.MinkowskiBatchNorm(self.out_dim) if self.use_bn else nn.Identity()
#        )
        self.gamma = nn.Sequential(
            nn.Conv1d(self.out_dim, self.hidden_dim, 1),
            nn.BatchNorm1d(self.hidden_dim) if self.use_bn else nn.Identity(),
            nn.ReLU(),
            nn.Conv1d(self.hidden_dim, self.vector_dim, 1),
            nn.BatchNorm1d(self.vector_dim) if self.use_bn else nn.Identity()
        )
        self.delta = nn.Sequential(
            nn.Conv2d(3, self.hidden_dim, 1),
            nn.BatchNorm2d(self.hidden_dim) if self.use_bn else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, self.out_dim, 1),
            nn.BatchNorm2d(self.out_dim) if self.use_bn else nn.Identity()
        )

        if self.pre_ln:
            self.ln_top = nn.LayerNorm(self.in_dim)
            self.ln_attn = nn.LayerNorm(self.hidden_dim)
            self.ln_down = nn.LayerNorm(self.out_dim)
            # @Niansong: Minkowski doesn't seem to have layer norm, I'm using InstanceNorm instead
            # self.in_top = ME.MinkowskiInstanceNorm(self.in_dim)
            # self.in_attn = ME.MinkowskiInstanceNorm(self.hidden_dim)
            # self.in_down = ME.MinkowskiInstanceNorm(self.out_dim)

        # TODO: what r should we use? I'm using 10 as default for now, perhaps too large for ModelNet40
        self.cube_query = cube_query(r=self.r, k=n_sample)


    def forward(self, x : ME.SparseTensor):
        '''
        input_p:  B, 3, npoint
        input_x: B, in_dim, npoint
        '''

        PT_begin = time.perf_counter()
        self.B = (x.C[:,0]).max().item() + 1 # batch size

        npoint, in_dim = tuple(x.F.size())
        k = min(self.n_sample, npoint)
        self.k = k
        h = self.nhead
        res = x

        # however the knn still gives 16 idxs
        # DEBUG: when using the r = 5, sometimes there is error
        if npoint <= self.n_sample:
            self.cube_query = cube_query(r=self.r, k=k)

        neighbor, mask = self.cube_query.get_neighbor(x, x)
        # neighbor: [B*npoint, k, bxyz]
        # mask: [B*npoint, k]
        
        if self.pre_ln:
            x = self.ln_top(x)

        x = self.linear_top(x) # [B, in_dim, npoint], such as [16, 32, 4096]

        if self.pre_ln:
            x = self.in_attn(x)

        # illustration on dimension notations:
        # - B: batch size
        # - nvoxel: number of all voxels of the whole batch
        # - k: k neighbors
        # - feat_dim: feature dimension, or channel as others call it
        # - nvoxel_batch: the maximum voxel number of a single SparseTensor in the current batch

        phi = self.phi(x).F # (nvoxel, feat_dim)
        phi = phi[:,None,:].repeat(1,k,1) # (nvoxel, k, feat_dim)

        psi = get_neighbor_feature(neighbor, self.psi(x)) # (nvoxel, k, feat_dim)
        # @Niansong: psi is the feature of each voxel's k neighbors
        alpha = get_neighbor_feature(neighbor, self.alpha(x)) # (nvoxel, k, feat_dim)
        


        try:
            relative_xyz = neighbor - x.C[:,None,:].repeat(1,k,1) # (nvoxel, k, bxyz), we later pad it to [B, xyz, nvoxel_batch, k]
        except RuntimeError:
            import ipdb; ipdb.set_trace()

        WITH_POSE_ENCODING = True
        if WITH_POSE_ENCODING:

            relative_xyz[:,0,0] = x.C[:,0] # get back the correct batch index, because we messed batch index in the subtraction above
            # since each batch could have different number of voxels, we need to pad them
            relative_xyz = pad_zero(relative_xyz, mask) # [B, xyz, nvoxel_batch, k]
            # @Niansong: a further illustration on batch_mask:
            # type: dict, content: {batch_idx : num of voxel}
            pose_encoding = self.delta(relative_xyz.float()) # (B, feat_dim, nvoxel_batch, k)
            # now we squeeze pose_encoding to (nvoxel, hidden_size, k), this should correspond to k SparseTensors
            # it is the positional encoding for each of the k neighbors

            time_begin = time.perf_counter()
            pose_tensor = make_position_tensor(pose_encoding, mask, x.C.shape[0]) # (nvoxel, k, feat_dim)

            time_end = time.perf_counter()
            # print('Overall took {}ms'.format((time_end - time_begin)*1e3))

        if self.use_vector_attn:
            # the attn_map: [vector_dim];
            # the alpha:    [out_dim]
            # attn_map = F.softmax(self.gamma(phi - psi + pos_encoding), dim=-1) # [B, in_dim, npoint, k], such as [16, 32, 4096, 16]
            # y = attn_map.repeat(1, self.out_dim // self.vector_dim,1,1)*(alpha + pos_encoding) # multiplies attention weight
            # self.out_dim and self.vector_dim are all 32 here, so y is still [16, 32, 4096, 16]
            # y = y.sum(dim=-1) # feature aggregation, y becomes [B, out_dim, npoint]

            if WITH_POSE_ENCODING:
                gamma_input = phi - psi + pose_tensor # (nvoxel, k, feat_dim)
            else:
                gamma_input = phi - psi # (nvoxel, k, feat_dim)
            gamma_input = gamma_input.permute(0, 2, 1) # (nvoxel, feat_dim, k)
            attn_map = F.softmax(self.gamma(gamma_input), dim=-1) # (nvoxel, feat_dim, k)
            if WITH_POSE_ENCODING:
                self_feat = (alpha + pose_tensor).permute(0,2,1) # (nvoxel, k, feat_dim) -> (nvoxel, feat_dim, k)
            else:
                self_feat = (alpha).permute(0,2,1) # (nvoxel, k, feat_dim) -> (nvoxel, feat_dim, k)
            y = attn_map.repeat(1, self.out_dim // self.vector_dim, 1, 1) * self_feat # (nvoxel, feat_dim, k)
            y = y.sum(dim=-1).view(x.C.shape[0], -1) # feature aggregation, y becomes (nvoxel, feat_dim)
            y = ME.SparseTensor(features = y, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)
        else:
            #phi = phi.reshape(B, h, self.out_dim//h, npoint, k)
            #psi = psi.reshape(B, h, self.out_dim//h, npoint, k)
            #attn_map = F.softmax((phi*psi).reshape(B, self.out_dim, npoint, k) + pos_encoding, dim=-1)
            #y = attn_map*(alpha+pos_encoding)
            #y = y.sum(dim=-1)
            pass

        
        if self.pre_ln:
            y = self.ln_down(y.transpose(1,2)).transpose(1,2)

        y = self.linear_down(y)

        PT_end = time.perf_counter()
        # print('PT blocks: {}ms'.format((PT_end - PT_begin)*1e3))

        return y+res, attn_map.detach().cpu().data


def make_position_tensor(pose_encoding : torch.Tensor, mask : torch.Tensor, nvoxel : int):
    """
    Mask positional encoding into k ME.SparseTensors

    Input:
        pose_encoding: (B, feat_dim, nvoxel_batch, k)
        batch_tensor:  (B, N)
    """
    # import ipdb; ipdb.set_trace()



    B, feat_dim, nvoxel_batch, k = pose_encoding.shape
    pose_encoding = pose_encoding.permute(0, 2, 3, 1) # (B, feat_dim, nvoxel_batch, k) -> (B, nvoxel_batch, k, feat_dim)

    masked_encoding = torch.zeros([nvoxel, k, feat_dim], device=pose_encoding.device).int()

    nums = mask.sum(-1)
    nvoxels = nums.repeat(nums.shape[0],1).tril().sum(dim=1)
    # nvoxels_leftshift = torch.cat([torch.zeros([]), nvoxels[:-1]])
    nvoxels_leftshift = nvoxels.roll(shifts=1)
    nvoxels_leftshift[0] = 0

    begin = time.perf_counter()
    torch.cuda.synchronize()
    end = time.perf_counter()

    '''
    pack the idxes to use the scatter/gather
    '''

    indexs_ = torch.cat([torch.arange(nums[i])+i*nvoxel_batch for i in range(len(nums))])

    import ipdb; ipdb.set_trace()

    for batch_idx in range(B):
        torch.cuda.synchronize()

        tick1 = time.perf_counter()

        # num = nums[batch_idx]
        masked_encoding[nvoxels_leftshift[batch_idx]:nvoxels[batch_idx], :, :] = pose_encoding[batch_idx, :nums[batch_idx], :, :] # @Niansong: feels wierd that we just throw away some feature
        # voxel_idx += num
        tick4 = time.perf_counter()
        # print('for-loop took {}ms'.format((tick4 - tick1)*1e3))

    print('Overall took {}ms'.format((end - begin)*1e3))


        # print('Overall took {}%'.format((tick3 - tick2)/(tick4 - tick1)))
    return masked_encoding # (nvoxel, k, feat_dim)

def get_neighbor_feature(neighbor: torch.Tensor, x: ME.SparseTensor):
    """
        fetch neighbor voxel's feature tensor.
        Input:
            neighbor: torch.Tensor [B*npoint, k, xyz]
            x:        ME.SparseTensor
    """
    B_npoint, k, _  = tuple(neighbor.size())
    neighbor = neighbor.view(-1, 4).float() # [B*npoint*k, bxyz]
    features = x.features_at_coordinates(neighbor)
    _, dim = features.shape
    features = features.view(-1, k, dim)
    return features

def pad_zero(tensor : torch.Tensor, mask: torch.Tensor):
    # input is [B*npoint, k, bxyz], we want [B, xyz, npoint, k]
    # need to pad zero because each batch may have different voxel number
    # B = int(max(tensor[:,0,0]).item() + 1)
    # k = tuple(tensor.shape)[1]
    B, N = mask.shape
    _, k, bxyz = tensor.shape
    result = torch.zeros([B, N, k, 4], dtype=torch.int, device=tensor.device)
    pointer = 0
    for b_idx in range(B):
        nvoxel = mask.sum(-1)[b_idx]
        result[b_idx, :nvoxel, :, :] = tensor[pointer:pointer+nvoxel, :, :]
        pointer += nvoxel
    result = result[:,:,:,1:] # (B, N, k, 3)
    result = result.permute(0, 3, 1, 2) # (B, N, k, 3) -> (B, 3, N, k)
    return result



def manhattan_dist(dxyz: tuple):
    dxyz = [abs(v) for v in dxyz]
    return sum(dxyz[1:])

def separate_batch(coord: torch.Tensor):
    """
        Input:
            coord: (N, 4) coordinate tensor, coord=b,x,y,z
        Return:
            tensor: (B, N, 3), batch index separated
            mask: (B, N), 1->valid, 0->invalid
    """
    B = (coord[:,0].max().item() + 1)
    # B = max(coord[:,0]).item() + 1

    batch_ids = coord[:,0]

    # get the splits of different i_batchA
    splits_at = torch.stack([torch.where(batch_ids == i)[0][-1] for i in torch.unique(batch_ids)]).int() # iter at i_batch_level
    splits_at_leftshift_one = torch.cat([torch.tensor([0.]).to(coord.device) , splits_at[:-1]], dim=0).int()
    len_per_batch = splits_at - splits_at_leftshift_one
    len_per_batch[0] = len_per_batch[0]+1 # DBEUG: stupid fix since 0~1566 has 1567 values
    N = len_per_batch.max().int().item()

    mask = torch.zeros(B, N, device=coord.device).long()
    new_coord = torch.zeros([B, N, 3]).int().to(coord.device) # (B, N, xyz)

    # TODO: maybe use torch.scatter could further boost speed here?
    for i_bs in range(B):
        if i_bs == 0:
            new_coord[i_bs][:len_per_batch[i_bs]] = coord[splits_at_leftshift_one[i_bs]:splits_at[i_bs]+1,1:]
            mask[i_bs][:len_per_batch[i_bs]] = 1
        else:
            new_coord[i_bs][:len_per_batch[i_bs]] = coord[splits_at_leftshift_one[i_bs]:splits_at[i_bs],1:]
            mask[i_bs][:len_per_batch[i_bs]] = 1


    # mask = (new_coord>0).int()[:,:,0]   # dirty fix of making [B,N,3] -> [B,N], since xyz should have same mask
    dat = new_coord

    '''
    voxeln = dict() # number of voxels of each object in the batch
    for b_idx in coord[:,0]:
        b_idx = int(b_idx)
        if not b_idx in voxeln: voxeln[b_idx] = 0
        voxeln[b_idx] += 1
    # TODO: since dict is on cpu, so slow, should fix em
    N = max(voxeln.values())
    # N = max(voxeln.items(), key=operator.itemgetter(1))[1]
    dat = torch.zeros([B, N, 3]).int().to(coord.device) # (B, N, xyz)
    mask   = torch.zeros([B, N]).int().to(coord.device) # (B, N)
    axis_idx = 0
    while axis_idx < coord.shape[0]:
        batch_idx = coord[axis_idx,0].item()
        num = voxeln[batch_idx]
        dat[batch_idx, 0:num, :] = torch.clone(coord[axis_idx:axis_idx+num, 1:])
        mask[batch_idx, 0:num] = 1
        axis_idx += num
    '''

    return dat, mask

def apply_coord_mask(indices, mask):
    """
        Input:
            indices: a tuple of three torch.Tensor (B-list, N-list, N-list)
            mask: torch.Tensor (B, N)
    """
    b_list = list()
    n1_list = list()
    n2_list = list()
    for idx in range(len(indices[0])):
        b = indices[0][idx].item()
        n1 = indices[1][idx].item()
        n2 = indices[2][idx].item()
        if mask[b][n1] == 0 or mask[b][n2] == 0: continue
        b_list.append(b)
        n1_list.append(n1)
        n2_list.append(n2)

    return (torch.tensor(b_list).long(), torch.tensor(n1_list).long(), torch.tensor(n2_list).long() )


# cube query for sparse tensors
class cube_query(object):
    """
    Cube query for ME.SparseTensor
        ref  : ME.SparseTensor, coord dim = [B * nr, 3]
            reference sparse tensor
        query: ME.SparseTensor, coord dim = [B * nq, 3]
            query sparse tensor, whose neighbors we are look for
    return:
        result: torch.Tensor [B * nq, k, 4], 4 is B,x,y,z
        mask:   torch.Tensor [B * nq, k], zero means less than k neighbors

    __init__():
        input:
            r: cube query radius
            k: k neighbors
    """
    def __init__(self, r, k):
        self.r = r
        self.k = k

    def get_neighbor(self, ref : ME.SparseTensor, query : ME.SparseTensor):

        # make ref's coord list a hash set
        # ref_set = set()
        # for i in range(B_nr):
            # coord_tuple = tuple([v.item() for v in ref.C[i] ] )
            # ref_set.add(coord_tuple)
        # build return tensorA

        B_nq, _ = query.C.shape

        # result = torch.zeros([B_nq, self.k, 4]).to(query.device)
        # mask   = torch.ones([B_nq, self.k]).to(query.device)
        # use torch tensor
        # directly operate on query.C cuda tensor
        # (nvoxel_q, 4) -> (nvoxel_q, 21*21*21, 4)
        # neighbors = query.C[:,None,:].repeat(1,(self.r*2+1)**3,1)
        # offset = torch.tensor([ (0,) + tuple(it) for it in itertools.product(range(-self.r,self.r+1), repeat=3) ]).to(query.C.device)
        # offset = offset[None,:,:].repeat(B_nq,1,1) # (nvoxel_q, cube_n, bxyz)
        # neighbors = neighbors + offset

        batch_begin = time.perf_counter()

        coord = query.C # (N, 4)
        coord, mask = separate_batch(coord) # (b, n, 3)
        b, n, _ = coord.shape

        batch_end = time.perf_counter()
        # print('Batch packing  tooks {}'.format((batch_end - batch_begin)*1e3))

        '''
        use pointnet++ like operation to acquire the idxes
        '''
        # time_begin = time.perf_counter()

        knn_begin = time.perf_counter()

        query_and_group_cuda = QueryAndGroup(radius=self.r, nsample=self.k, use_xyz=False)
        coord = coord.float()
        idxs = query_and_group_cuda(
            xyz=coord,
            new_xyz=coord,
            features=coord.transpose(1,2).contiguous(),
        ) # idx: [bs, xyz, npoint, nsample]
        idxs = idxs.permute([0,2,3,1]) # idx: [bs, npoint, nsample, xyz]
        result_padded = idxs

        knn_end = time.perf_counter()
        # print('KNN  tooks {}'.format((knn_end - knn_begin)*1e3))

        # time_end = time.perf_counter()
        # print('took {} ms'.format((time_end - time_begin)*1e3))


        # extended_coord = coord.unsqueeze(2).repeat(1,1,coord.shape[1],1)
        # diff = torch.abs(coord.unsqueeze(2) - coord.unsqueeze(1)) # (b, n, n, 3)
        # # (b, n, 1, 3) - (b, 1, n, 3)
        # diff = diff.sum(dim=-1)
        # indices = torch.argsort(diff, dim=-1)[:,:,:self.k] # (b,n,k)

        '''
        older_version

        indices = torch.where(diff <= self.r) # we can reshape it to (b, n, n)
        #masked_indices = apply_coord_mask(indices, mask)
        # neighbors = coord[indices]

        # if there's not enough k neighbors, we fill with random neighbors
        # centers = coord[masked_indices[0], masked_indices[1]] # (pair number, 3)
        # neighbors = coord[masked_indices[0], masked_indices[2]] # (pair number, 3)

        # TODO: FIX here, maybe cant reshape
        result_padded = torch.zeros([b, n, self.k, 3]).int().to(diff.device) # (b, n, k, 3)
        neighbors = coord[indices[0], indices[2]].reshape(diff.shape[0], diff.shape[1], diff.shape[2], 3) # (b, n, n, 3)
        if min(mask.sum(dim=-1)) >= self.k:
            # if there are guaranteed more than k voxels in the object
            result_padded = neighbors[:,:,self.k,:]
        else:
            # if there are less than k voxels in the object, we have to fill k neighbors with repeated voxels
            # we have to loop through batch because each object has different voxel number
            for b in range(diff.shape[0]):
                n = mask.sum(dim=-1)[b]
                pointer = 0
                while pointer < self.k:
                    end = min(pointer + n, self.k)
                    result_padded[b,:,pointer:end,:] = neighbors[b,:,:min(n, self.k-pointer),:]
                    pointer += n
        '''
        # unpad result (b, n, k, 3) -> (B_nq, k, 4) by applying mask
        pack_begin = time.perf_counter()

        result = torch.zeros([B_nq, self.k, 4], dtype=torch.int32, device=query.device)
        pointer = 0
        for b_idx in range(result_padded.shape[0]):
            n = mask.sum(dim=-1)[b_idx]
            result[pointer:pointer+n, :, 1:] = result_padded[b_idx,:n, :, :]
            result[pointer:pointer+n, :, 0]  = b_idx
            pointer += n

        pack_end = time.perf_counter()
        # print('Pack tooks {}'.format((pack_end - pack_begin)*1e3))

#        for i in range(B_nq):
#            # neighborhood
#            # n, x, y, z = query.C[i][0].item(), query.C[i][1].item(), query.C[i][2].item(), query.C[i][3].item()
#            n, x, y, z = [v.item() for v in query.C[i]]
#            neighbor = list()
##            offset_loop_start = time.perf_counter()
##            hash_time = 0
#            # we need to optimize this loop, it costs around 10 ms
##            for offset in itertools.product(range(-self.r, self.r+1), repeat=3):
##                if offset == (0,0,0): continue # skip center
##                nb = tuple(sum(x) for x in zip(offset, (x,y,z))) # a neighbor voxel's coordinate
##                nb = (n, *nb) # don't forget batch index          
##                hash_start = time.perf_counter()
##                if nb in ref_set: neighbor.append((nb, manhattan_dist(offset) ) ) # a tuple: (coord, manhattan distance)
##                hash_end = time.perf_counter()
##                hash_time += hash_end - hash_start
##            offset_loop_end = time.perf_counter()
##            print(f"offset loop : {(offset_loop_end - offset_loop_start)*1e3} ms") # about 13 ms
##            print(f"    hash time: {hash_time*1e3} ms") # about 1 ms
#
#            # try vectorization with numpy
#            offset = np.mgrid[-self.r:self.r+1, -self.r:self.r+1, -self.r:self.r+1]
#            offset[0] += x; offset[1] += y; offset[2] += z
#            offset = np.transpose(offset, (1,2,3,0))
#            vec_start = time.perf_counter()
#            # offset (20,20,20,3) 
#            cond = [(n,) + tuple(offset[it]) in ref_set for it in itertools.product(range(-self.r,self.r+1), repeat=3)]
#            # this condition building is slow ~13ms
#            vec_end = time.perf_counter()
#            cond = np.reshape(cond, (2*self.r+1, 2*self.r+1, 2*self.r+1))
#            masked = np.where(cond)
#            nb_array = offset[masked[0], masked[1], masked[2]]
#            print(f"vectorized mask time : {(vec_end-vec_start)*1e3}")
#            for idx in range(nb_array.shape[0]):
#                nb = (n,) + tuple(nb_array[idx,:])
#                neighbor.append((nb, manhattan_dist((nb[0], nb[1]-x, nb[2]-y, nb[3]-z)) ) )
#            # sort neighbor according to manhattan distance
#            neighbor = sorted(neighbor, key=lambda t : t[1])
#            for k_idx in range(self.k):
#                if k_idx > len(neighbor)-1 :
#                    mask[i][k_idx] = 0
#                    continue
#                bxyz = neighbor[k_idx][0]
#                result[i][k_idx] = torch.tensor([*bxyz])
        return result, mask

if __name__ == "__main__":
    import torch
    import MinkowskiEngine as ME
    feature = torch.tensor([[0.2, 0.3], [0.4, 0.5]])
    coord =torch.tensor([[0.6, 0.8, 0.3], [0.4, 0.3, 0.5]])
    x = ME.SparseTensor(
        features = feature,
        coordinates = ME.utils.batched_coordinates([coord / 0.1])
    )
    cq = cube_query(1, 2)
    result, mask = cq.get_neighbor(x, x)
