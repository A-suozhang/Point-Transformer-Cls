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


def sample_and_group_cuda(npoint, k, xyz, points, instance=None, instance_relation=None):
    """
    Input:
        npoint: seems 1/4 of N
        k:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, C, N]
        instance: input_instance, [B,N]
    Return:
        new_xyz: sampled points position data, [B, 3, npoint]
        new_points: sampled points data, [B, C+C_xyz, npoint, k]
        grouped_xyz_norm: sampled relative points position data, [B, 3, npoint, k]
        new_instance, [B, npoint]
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

    # unsqueeze to [B,N,1] then apply indexing
    if instance is not None:
        new_instance = index_points_cuda(instance.unsqueeze(-1).float(), fps_idx).squeeze(-1)
    else:
        pass

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
    
    if instance is not None:
        return new_xyz.transpose(1,2), grouped_xyz_norm, new_points, new_instance
    else:
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

    def forward(self, xyz, points, instance=None, instance_relation=None):
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
    
        if instance is not None:
            new_xyz, grouped_xyz_norm, new_points, new_instance = sample_and_group_cuda(self.npoint, self.k, xyz, points, instance)
        else:
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
        if instance is not None:
            return new_xyz, new_points_pooled, grouped_xyz_norm, new_points, new_instance
        else:
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

    def forward(self, input_p, input_x, instance=None, instance_relation=None):
        '''
        input_p:  B, 3, npoint
        input_x: B, in_dim, npoint
        '''


        '''
        how to use the instance information:
        1. use it as guidance of the attention, mask the knns points with different instance label
        2. directly random choose points of same instance label as attention receptive field
        3. attend to the instance center
        '''
        INSTANCE_SCHEME = 3

        B, in_dim, npoint = list(input_x.size())
        n_sample = self.n_sample
        k = min(n_sample, npoint)
        h = self.nhead

        res = input_x

        input_p = input_p.permute([0,2,1])
        ori_input_p = input_p


        if instance is not None and INSTANCE_SCHEME == 1:
            # knn more points for sampling
            knn_sample_more_ratio = 2
            enlarged_k = k*knn_sample_more_ratio
            self.knn = KNN(k=enlarged_k, transpose_mode=True)
        else:
            self.knn = KNN(k=k, transpose_mode=True)

        # DEBUG: error here is that in the last block only 4-points;
        # however the knn still gives 16 idxs
        # so when n-point is smaller than the k(n_smaple)
        # if npoint < self.n_sample:
            # self.knn = KNN(k=npoint, transpose_mode=True) 
        # else:
            # self.knn = KNN(k=n_sample, transpose_mode=True)
            # pass # regular case


        # DEBUG ONLY: using the input_x: feature space knn!
        # _, idx = self.knn(input_x.transpose(1,2), input_x.transpose(1,2))


        if instance is not None:

            if INSTANCE_SCHEME == 3:
                '''
                Ver3.0: use cur instance center as knn center,
                calc the instance center, and weighting the cur-idx and the instance center idx
                ERROR:
                    - all points of the same instance will have the same idxes? and all are cloest N points to centroid
                    - if use weiighted center and coord, However, need to do N-pointx KNN, will be slow...
                '''
                ori_input_p = input_p.clone()
                # where = torch.where(instance[0] == 1)
                # instance_xyz = input_p[:,where,:].mean(dim=1)   # get [bs, 3] centroid for cur instance
                for i_bs in range(instance.shape[0]):
                    for v in torch.unique(instance[i_bs]):
                        tmp_idx = torch.where(instance[i_bs] == v)[0]
                        ins_center = input_p[:, tmp_idx, :].mean(dim=1) # the centroids for each intsance
                        # average cur point and the instance center
                        alpha = 0.999
                        input_p[:,tmp_idx,:] = alpha*input_p[:,tmp_idx,:] + (1-alpha)*ins_center.unsqueeze(1) # [bs, n_cur_ins, 3] + [bs, 1, 3]


                _, idx = self.knn(ori_input_p.contiguous(), ori_input_p)
                _, idx2 = self.knn(ori_input_p.contiguous(), input_p)
                print( (idx == idx2).int().sum() / idx.nelement())
            else:
                _, idx = self.knn(input_p.contiguous(), input_p)
        else:
            _, idx = self.knn(input_p.contiguous(), input_p)

        idx = idx.int()

        if INSTANCE_SCHEME == 1:
            '''
            Ver1.0(Naive Version): mask the knn(instance label as auxiliary filter)
            older version of the instance mask
            directly ck if knn grouped point within the same pointset
            then mask if not in
            '''

            if instance is not None:
                # print('start processing the instance mask')
                masks = []
                for i_bs, idx_cur_bs in enumerate(idx):
                    # [4096, 16] cur_bs_idx
                    # [4096]: instance_label[i_bs]
                    mask = instance[i_bs][idx_cur_bs.long()] # [4096, 2*k]
                    mask = mask - mask[:,0].unsqueeze(-1) # get the 1st column(the 1st element in k-elements is itself)
                    mask = (mask == 0).int() # acuiqre the 0-1 mask
                    masks.append(mask)
                masks = torch.stack(masks)
                print("mask ratio {:.4f}".format(masks.sum() / masks.nelement())) # >0.5 means ok

                '''
                generate bigger knn-idx and mask, then choose the 1st n_sample(16) elements
                random sample other points from the latter, and use mask to fill into the 0 ones

                get the 1st k idxes that is not 0 in mask
                since the mask values are all 0-1, use argsort will return a vector
                however, we want smaller idxes in the front
                so we give 0 elments a large value to make it appears at last
                if use descend=True, biggest idx with 1 will come first
                '''

                inv_masks = (masks == 0).int()
                tmp_inds = torch.arange(masks.shape[2]).repeat(masks.shape[0],masks.shape[1],1).to(idx.device) # generate the [1,2,...,enlarged_k] inds
                tmp_inds = tmp_inds*masks
                tmp_inds = tmp_inds + (masks.shape[2]+1)*inv_masks # fill the places of 0 with value bigger than the maximum value
                tmp_inds = torch.argsort(tmp_inds)[:,:,:k]  # after argsort, the former elements should be non-zero value with smaller idx
                idx = torch.gather(idx, -1, tmp_inds)
                idx = idx.int()

                # TODO: if nk still does not contain enough elements, the argsort will contain the closet knn result while not instance


        elif INSTANCE_SCHEME == 2:

            '''
            # Ver2.0: directly use the points of the same instance label as neighbors
            # random sample k points in the same instance
            '''
            if instance is not None:
                instance_relations = []
                for i_bs in range(instance.shape[0]):
                    instance_inds = [torch.where(instance[i_bs] == v)[0] for v in torch.unique(instance[i_bs])] # torch.where returns a tuple, so use [0] to getthe tensor
                    instance_relation = torch.full([instance[0].shape[0], k], -1).to(instance.device)
                    for i, ins_id in enumerate(instance_inds):
                        # TODO; stupid pytorch has no func like random.choice 
                        if len(ins_id) <= 5: # for small outlier points, skip em
                            continue
                        try:
                            perms = torch.multinomial(ins_id.repeat(len(ins_id),1).float(), num_samples=min(k, len(ins_id)), replacement=False)
                        except RuntimeError:
                            import ipdb; ipdb.set_trace()
                        choices = ins_id[perms]
                        instance_relation[instance_inds[i],:choices.shape[1]] = choices
                    instance_relation[:,0] = torch.arange(instance_relation.shape[0])
                    instance_relations.append(instance_relation)
                instance_relations = torch.stack(instance_relations)

                # print('replacing the instance_relation')
                instance_relation_nonzero_mask = (instance_relations>=0).int()
                instance_relation_zero_mask = (instance_relations<0).int()

                idx = idx*instance_relation_zero_mask + instance_relations*instance_relation_nonzero_mask
                idx = idx.int()

        # ===================== Deprecated Methods ===========================1

        '''
        # Ver 2.3: failed version of receiving a instance_relation,
        # however, point downsample could not be handled
        # the instance feed in here is of the same size as the idxes
        # if the num within the same instance group as less than k
        # then the instance_relation will contain -1, we will replace these -1s
        # with the original idx acquired by knn
        if instance_relation is not None:
            print('replacing the instance_relation')
            # import ipdb; ipdb.set_trace()

            instance_relation = instance_relation[:,:,:k]
            instance_relation_nonzero_mask = (instance_relation>=0).int()
            instance_relation_zero_mask = (instance_relation<0).int()

            # idx = idx*instance_relation_zero_mask + instance_relation*instance_relation_nonzero_mask
            idx = instance_relation.int()

        '''

        '''
        Ver 2.2: Hash Table-based
        1st pack the instance into dict(hash table)
        then ref the points within the same scope to replace the knn points
        if ont enough points of the same insatcne, keep the knn idxs
        '''
        '''
        # pack the instacne into dict for further reference
        if instance is not None:
            print('start creating instance dicts')
            instance_dicts = []
            for i_bs, instance_cur_bs in enumerate(instance):
                instance_dict = {}
                for ins_idx, ins in enumerate(instance_cur_bs):
                    if ins.item() in instance_dict.keys():
                        instance_dict[ins.item()].append(ins_idx)
                    else:
                        instance_dict[ins.item()] = [ins_idx]
                for ins_k in instance_dict.keys():
                    instance_dict[ins_k] = torch.tensor(instance_dict[ins_k]).to(instance.device)
                instance_dicts.append(instance_dict)

            l1 = []
            for i_bs in range(instance.shape[0]):
                l0 = []
                for i_point in range(instance.shape[1]):
                    tmp = torch.zeros([k])
                    instance_gathered  = instance_dicts[i_bs][instance[i_bs][i_point].item()][:k]
                    tmp[:len(instance_gathered)] = instance_gathered
                    # idx[i_bs][i_point][:len(instance_gathered)] = instance_gathered
                    l0.append(tmp)
                tmp1 = torch.stack(l0)
                l1.append(tmp1)
            new_idx = torch.stack(l1)
        '''

        '''
        Ver: 2.1: Naive Version of for-loop replacement 
        # Too slow version, needs improving
        # 1st use knn then use mask the value belongs not to the same instance
        instance_masks = []
        for i_batch, single_batch_instance in enumerate(instance):
            # single_batch_instance: [npoint]
            masks_cur_batch = []
            for i_point, gathered_points in enumerate(idx[i_batch]):
                # gathered_points: [k]
                points_with_same_instance = torch.where(single_batch_instance == single_batch_instance[i_point])[0]
                # ck if the grouped idxes are within the same idxes
                cur_mask = torch.tensor([g.item() in points_with_same_instance for g in gathered_points])
                masks_cur_batch.append(cur_mask)
            masks_cur_batch = torch.stack(masks_cur_batch)
            instance_masks.append(masks_cur_batch)
        instance_masks = torch.stack(instance_masks)
        '''

        # ==========================================================================================

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
            attn_map = F.softmax(self.gamma(phi - psi + pos_encoding), dim=-1) # [B, Dim, N, k]
            # if instance is not None: # apply mask
                # attn_map = attn_map*(masks.unsqueeze(1))
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

