import pickle
import os
import sys
import numpy as np
import torch
import torch.utils.data as torch_data
from torch.utils.data import DataLoader

class ScannetDataset(torch_data.Dataset):
    def __init__(self, root=None, npoints=10240, split='train', with_dropout=False, with_norm=True, with_rgb=True, with_instance=False,sample_rate=None, k=16):
        super().__init__()
        print(' ---- load data from', root)
        self.npoints = npoints
        self.with_dropout = with_dropout

        self.indices = [0, 1, 2]
        if with_norm: self.indices += [3, 4, 5]
        if with_rgb: self.indices += [6, 7, 8]
        self.with_instance = with_instance

        # TODO: dirty, should fix
        self.k = k # k is the max neighbor k in point-transformer arch

        print('load scannet dataset <{}> with npoint {}, indices: {}.'.format(split, npoints, self.indices))

        # deprecated version of pickle load
        # data_filename = os.path.join(root, 'scannet_%s_rgb21c_pointid.pickle' % (split))
        # with open(data_filename, 'rb') as fp:
            # self.scene_points_list = pickle.load(fp)
            # self.semantic_labels_list = pickle.load(fp)
            # # scene_points_id = pickle.load(fp)
            # num_point_all = pickle.load(fp)

        # TEST: newer loading of the pth file
        data_filename = os.path.join(root, 'new_{}.pth'.format(split))
        data_dict = torch.load(data_filename)
        self.scene_points_list = data_dict['data']
        self.semantic_labels_list = data_dict['label']
        if self.with_instance:
            self.instance_label_list = data_dict['instance']
        #scene_points_id = pickle.load(fp)
        num_point_all = data_dict['npoints']

        if split == 'train':
            labelweights = np.zeros(21)
            for seg in self.semantic_labels_list:
                tmp,_ = np.histogram(seg,range(22))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            # self.labelweights = 1/np.log(1.2+labelweights)
            self.labelweights = np.power(np.amax(labelweights[1:]) / labelweights, 1 / 3.0)

        elif split == 'eval' or split == 'test':
            self.labelweights = np.ones(21)
        else:
            raise ValueError('split must be train or eval.')

        # sample & repeat scenes, older version deprecated
        if sample_rate is not None:
            num_point = npoints
            sample_prob = num_point_all / np.sum(num_point_all)
            num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
            room_idxs = []
            for index in range(len(self.scene_points_list)):
                repeat_times = round(sample_prob[index] * num_iter)
                repeat_times = int(max(repeat_times, 1))
                room_idxs.extend([index] * repeat_times)
            self.room_idxs = np.array(room_idxs)
            np.random.seed(123)
            np.random.shuffle(self.room_idxs)
        else:
            self.room_idxs = np.arange(len(self.scene_points_list))

        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __getitem__(self, index):
        index = self.room_idxs[index]
        data_set = self.scene_points_list[index]
        point_set = data_set[:, :3]
        if self.with_instance:
            instance_set = self.instance_label_list[index]
        semantic_seg = self.semantic_labels_list[index].astype(np.int32)
        coordmax = np.max(point_set, axis=0)
        coordmin = np.min(point_set, axis=0)
        mpmin = np.maximum(coordmax-[2, 2, 3.0], coordmin)
        smpmin[2] = coordmin[2]
        smpsz = np.minimum(coordmax-smpmin,[2,2,3.0])
        smpsz[2] = coordmax[2]-coordmin[2]
        isvalid = False
        # randomly choose a point as center point and sample <n_points> points in the box area of center-point
        for i in range(10):
            curcenter = point_set[np.random.choice(len(semantic_seg),1)[0],:]
            curmin = curcenter - [1, 1, 1.5]
            curmax = curcenter + [1, 1, 1.5]
            curmin[2] = coordmin[2]
            curmax[2] = coordmax[2]
            curchoice = np.sum((point_set >= (curmin - 0.2)) * (point_set <= (curmax + 0.2)), axis=1) == 3
            cur_point_set = point_set[curchoice, :]
            cur_data_set = data_set[curchoice, :]
            if self.with_instance:
                try:
                    cur_instance_set = instance_set[curchoice]
                except IndexError:
                    import ipdb; ipdb.set_trace()
            cur_semantic_seg = semantic_seg[curchoice]
            if len(cur_semantic_seg) == 0:
                continue
            mask = np.sum((cur_point_set >= (curmin - 0.01)) * (cur_point_set <= (curmax + 0.01)), axis=1) == 3
            vidx = np.ceil((cur_point_set[mask, :] - curmin) / (curmax - curmin) * [31.0, 31.0, 62.0])
            vidx = np.unique(vidx[:, 0] * 31.0 * 62.0 + vidx[:, 1] * 62.0 + vidx[:, 2])
            isvalid = np.sum(cur_semantic_seg > 0) / len(cur_semantic_seg) >= 0.7 and len(vidx) / 31.0 / 31.0 / 62.0 >= 0.02
            if isvalid:
                break

        choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)
        semantic_seg = cur_semantic_seg[choice]
        if self.with_instance:
            instance_seg = cur_instance_set[choice]
        mask = mask[choice]
        sample_weight = self.labelweights[semantic_seg]
        sample_weight *= mask

        selected_points = cur_data_set[choice, :] # np * 6, xyz + rgb
        point_set = np.zeros((self.npoints, 9)) # xyz, norm_xyz, rgb

        point_set[:, :3] = selected_points[:, :3] # xyz
        for i in range(3): # normalized_xyz
            point_set[:, 3 + i] = (selected_points[:, i] - coordmin[i]) / (coordmax[i] - coordmin[i])
        point_set[:, 6:] = selected_points[:, 3:] / 255.0 # rgb

        if self.with_dropout:
            dropout_ratio = np.random.random() * 0.875 # 0 ~ 0.875
            drop_idx = np.where(np.random.random((self.npoints)) <= dropout_ratio)[0]

            point_set[drop_idx, :] = point_set[0, :]
            semantic_seg[drop_idx] = semantic_seg[0]
            sample_weight[drop_idx] *= 0

        point_set = point_set[:, self.indices]

        # Pack the instance
        # WARNING: the deprecated(failed attempt) of the instance_relatiion dict
        # if self.with_instance:
            # k = self.k
            # idxes = [np.where(instance_seg == x)[0] for x in np.unique(instance_seg)]
            # instance_relations = np.full([instance_seg.size,k], -1)
            # for i, idx in enumerate(idxes):
                # choices = np.random.choice(idxes[i], (idxes[i].size,k))
                # instance_relations[idxes[i]] = choices
            # instance_relations[:,0] = np.arange(instance_relations.shape[0])
            # instance_relations = instance_relations.astype(int)

        if self.with_instance:
            return point_set, semantic_seg, sample_weight, instance_seg
        else:
            return point_set, semantic_seg, sample_weight


    def __len__(self):
        return len(self.room_idxs)
        # return len(self.scene_points_list)

class ScannetDatasetWholeScene(torch_data.IterableDataset):
    def __init__(self, root=None, npoints=10240, split='train', with_norm=True, with_rgb=True):
        super().__init__()
        print(' ---- load data from', root)
        self.npoints = npoints
        
        self.indices = [0, 1, 2]
        if with_norm: self.indices += [3, 4, 5]
        if with_rgb: self.indices += [6, 7, 8]
        print('load scannet <whole scene> dataset <{}> with npoint {}, indices: {}.'.format(split, npoints, self.indices))
        
        self.temp_data = []
        self.temp_index = 0
        self.now_index = 0
        
        data_filename = os.path.join(root, 'scannet_%s_rgb21c_pointid.pickle' % (split))
        with open(data_filename, 'rb') as fp:
            self.scene_points_list = pickle.load(fp)
            self.semantic_labels_list = pickle.load(fp)
        if split == 'train':
            labelweights = np.zeros(21)
            for seg in self.semantic_labels_list:
                tmp,_ = np.histogram(seg,range(22))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            # self.labelweights = 1 / np.log(1.2 + labelweights)
            self.labelweights = np.power(np.amax(labelweights[1:]) / labelweights, 1 / 3.0)
        elif split == 'eval' or split == 'test':
            self.labelweights = np.ones(21)
    
    def get_data(self):
        idx = self.temp_index
        self.temp_index += 1
        return self.temp_data[idx]

    def reset(self):
        self.temp_data = []
        self.temp_index = 0
        self.now_index = 0

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        if self.now_index >= len(self.scene_points_list) and self.temp_index >= len(self.temp_data):
            raise StopIteration()

        if self.temp_index < len(self.temp_data):
            return self.get_data()

        index = self.now_index
        self.now_index += 1
        self.temp_data = []
        self.temp_index = 0

        # print(self.temp_index, self.now_index, len(self.scene_points_list))

        data_set_ini = self.scene_points_list[index]
        point_set_ini = data_set_ini[:,:3]
        semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)
        coordmax = np.max(point_set_ini,axis=0)
        coordmin = np.min(point_set_ini,axis=0)
        grid_size=2
        nsubvolume_x = np.ceil((coordmax[0]-coordmin[0])/grid_size).astype(np.int32)
        nsubvolume_y = np.ceil((coordmax[1]-coordmin[1])/grid_size).astype(np.int32)
        point_sets = list()
        semantic_segs = list()
        sample_weights = list()
        isvalid = False
        for i in range(nsubvolume_x):
            for j in range(nsubvolume_y):
                curmin = coordmin+[i*grid_size,j*grid_size,0]
                curmax = coordmin+[(i+1)*grid_size,(j+1)*grid_size,coordmax[2]-coordmin[2]]
                curchoice = np.sum((point_set_ini>=(curmin-0.2))*(point_set_ini<=(curmax+0.2)),axis=1)==3
                cur_point_set = point_set_ini[curchoice,:]
                cur_data_set = data_set_ini[curchoice,:]
                cur_semantic_seg = semantic_seg_ini[curchoice]
                if len(cur_semantic_seg)==0:
                    continue
                mask = np.sum((cur_point_set >= (curmin - 0.001)) * (cur_point_set <= (curmax + 0.001)), axis=1) == 3
                
                choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=len(cur_semantic_seg) < self.npoints)
                semantic_seg = cur_semantic_seg[choice] # N
                mask = mask[choice]
                if sum(mask) / float(len(mask)) < 0.01:
                    continue
                sample_weight = self.labelweights[semantic_seg]
                sample_weight *= mask # N
                
                selected_points = cur_data_set[choice, :] # Nx6
                point_set = np.zeros([self.npoints, 9])
                point_set[:, :3] = selected_points[:, :3] # xyz
                for k in range(3): # normalized_xyz
                    point_set[:, 3 + k] = (selected_points[:, k] - coordmin[k]) / (coordmax[k] - coordmin[k])
                point_set[:, 6:] = selected_points[:, 3:] / 255.0 # rgb

                point_set = point_set[:, self.indices]
                self.temp_data.append((point_set, semantic_seg, sample_weight))

        return self.get_data()

class ScannetDatasetWholeScene_evaluation(torch_data.IterableDataset):
    #prepare to give prediction on each points
    def __init__(self, root=None, scene_list_dir=None, split='test', num_class=21, block_points=81932, with_norm=True, with_rgb=True):
        super().__init__()
        print(' ---- load data from', root)
        self.block_points = block_points
        self.indices = [0, 1, 2]
        if with_norm: self.indices += [3, 4, 5]
        if with_rgb: self.indices += [6, 7, 8]
        print('load scannet <TEST> dataset <{}> with npoint {}, indices: {}.'.format(split, block_points, self.indices))
        
        self.point_num = []
        self.temp_data = []
        self.temp_index = 0
        self.now_index = 0
        
        data_filename = os.path.join(root, 'scannet_%s_rgb21c_pointid.pickle' % (split))
        with open(data_filename, 'rb') as fp:
            self.scene_points_list = pickle.load(fp)
            self.semantic_labels_list = pickle.load(fp)
            # self.scene_points_id = pickle.load(fp)
            self.scene_points_num = pickle.load(fp)
            file_path = os.path.join(scene_list_dir, 'scannetv2_{}.txt'.format(split))

        num_class = 21
        if split == 'test' or split == 'eval' or split == 'train':
            self.labelweights = np.ones(num_class)
            for seg in self.semantic_labels_list:
                self.point_num.append(seg.shape[0])
            
            with open(file_path) as fl:
                self.scene_list = fl.read().splitlines()
        else:
            raise ValueError('split must be test or eval, {}'.format(split))

    def reset(self):
        self.temp_data = []
        self.temp_index = 0
        self.now_index = 0

    def __iter__(self):
        if self.now_index >= len(self.scene_points_list):
            print(' ==== reset dataset index ==== ')
            self.reset()
        self.gen_batch_data()
        return self

    def chunks(self, l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]
    
    def split_data(self, data, idx):
        new_data = []
        for i in range(len(idx)):
            new_data += [data[idx[i]]]
        return new_data
    
    def nearest_dist(self, block_center, block_center_list):
        num_blocks = len(block_center_list)
        dist = np.zeros(num_blocks)
        for i in range(num_blocks):
            dist[i] = np.linalg.norm(block_center_list[i] - block_center, ord = 2) #i->j
        return np.argsort(dist)[0]

    def gen_batch_data(self):
        index = self.now_index
        self.now_index += 1
        self.temp_data = []
        self.temp_index = 0

        print(' ==== generate batch data of {} ==== '.format(self.scene_list[index]))

        delta = 1.0
        # delta = 4.0
        # if self.with_rgb:
        point_set_ini = self.scene_points_list[index]
        # else:
        #     point_set_ini = self.scene_points_list[index][:, 0:3]
        semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)
        coordmax = np.max(point_set_ini[:, 0:3],axis=0)
        coordmin = np.min(point_set_ini[:, 0:3],axis=0)
        nsubvolume_x = np.ceil((coordmax[0]-coordmin[0])/delta).astype(np.int32)
        nsubvolume_y = np.ceil((coordmax[1]-coordmin[1])/delta).astype(np.int32)
        point_sets = []
        semantic_segs = []
        sample_weights = []
        point_idxs = []
        block_center = []
        for i in range(nsubvolume_x):
            for j in range(nsubvolume_y):
                curmin = coordmin+[i*delta,j*delta,0]
                curmax = curmin+[2,2,coordmax[2]-coordmin[2]]
                curchoice = np.sum((point_set_ini[:,0:3]>=(curmin-0.2))*(point_set_ini[:,0:3]<=(curmax+0.2)),axis=1)==3
                curchoice_idx = np.where(curchoice)[0]
                cur_point_set = point_set_ini[curchoice,:]
                cur_semantic_seg = semantic_seg_ini[curchoice]
                if len(cur_semantic_seg)==0:
                    continue
                mask = np.sum((cur_point_set[:,0:3]>=(curmin-0.001))*(cur_point_set[:,0:3]<=(curmax+0.001)),axis=1)==3
                sample_weight = self.labelweights[cur_semantic_seg]
                sample_weight *= mask # N
                point_sets.append(cur_point_set) # 1xNx3/6
                semantic_segs.append(cur_semantic_seg) # 1xN
                sample_weights.append(sample_weight) # 1xN
                point_idxs.append(curchoice_idx) #1xN
                block_center.append((curmin[0:2] + curmax[0:2]) / 2.0)

        # merge small blocks
        num_blocks = len(point_sets)
        block_idx = 0
        while block_idx < num_blocks:
            if point_sets[block_idx].shape[0] > self.block_points // 2:
                block_idx += 1
                continue

            small_block_data = point_sets[block_idx].copy()
            small_block_seg = semantic_segs[block_idx].copy()
            small_block_smpw = sample_weights[block_idx].copy()
            small_block_idxs = point_idxs[block_idx].copy()
            small_block_center = block_center[block_idx].copy()
            point_sets.pop(block_idx)
            semantic_segs.pop(block_idx)
            sample_weights.pop(block_idx)
            point_idxs.pop(block_idx)
            block_center.pop(block_idx)
            nearest_block_idx = self.nearest_dist(small_block_center, block_center)
            point_sets[nearest_block_idx] = np.concatenate((point_sets[nearest_block_idx], small_block_data), axis = 0)
            semantic_segs[nearest_block_idx] = np.concatenate((semantic_segs[nearest_block_idx], small_block_seg), axis = 0)
            sample_weights[nearest_block_idx] = np.concatenate((sample_weights[nearest_block_idx], small_block_smpw), axis = 0)
            point_idxs[nearest_block_idx] = np.concatenate((point_idxs[nearest_block_idx], small_block_idxs), axis = 0)
            num_blocks = len(point_sets)

        # divide large blocks
        num_blocks = len(point_sets)
        div_blocks = []
        div_blocks_seg = []
        div_blocks_smpw = []
        div_blocks_idxs = []
        div_blocks_center = []
        for block_idx in range(num_blocks):
            cur_num_pts = point_sets[block_idx].shape[0]

            point_idx_block = np.array([x for x in range(cur_num_pts)])
            # if could not be divided by num_point
            # random fecth points within the set
            if point_idx_block.shape[0]%self.block_points != 0:
                makeup_num = self.block_points - point_idx_block.shape[0]%self.block_points
                np.random.shuffle(point_idx_block)
                point_idx_block = np.concatenate((point_idx_block,point_idx_block[0:makeup_num].copy()))

            np.random.shuffle(point_idx_block)

            # split into Nxnpoint
            sub_blocks = list(self.chunks(point_idx_block, self.block_points))

            div_blocks += self.split_data(point_sets[block_idx], sub_blocks)
            div_blocks_seg += self.split_data(semantic_segs[block_idx], sub_blocks)
            div_blocks_smpw += self.split_data(sample_weights[block_idx], sub_blocks)
            div_blocks_idxs += self.split_data(point_idxs[block_idx], sub_blocks)
            div_blocks_center += [block_center[block_idx].copy() for i in range(len(sub_blocks))]

        for i in range(len(div_blocks)):
            selected_points = div_blocks[i]
            point_set = np.zeros([self.block_points, 9])
            point_set[:, :3] = selected_points[:, :3] # xyz
            for k in range(3): # normalized_xyz
                point_set[:, 3 + k] = (selected_points[:, k] - coordmin[k]) / (coordmax[k] - coordmin[k])
            point_set[:, 6:] = selected_points[:, 3:] / 255.0 # rgb

            point_set = point_set[:, self.indices]
            self.temp_data.append((point_set, div_blocks_seg[i], div_blocks_smpw[i], div_blocks_idxs[i]))


    def __next__(self):
        if self.temp_index >= len(self.temp_data):
            raise StopIteration()
        else:
            idx = self.temp_index
            self.temp_index += 1
            return self.temp_data[idx]


if __name__  == "__main__":
    # DEFAULT using only xyz, since with_rgb and with_norm is False
    # TODO: change it to make it normally use all 9-dims
    trainset = ScannetDataset(root='../data/scannet_v2/scannet_pickles', npoints=8192, split='train', with_instance=True)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True)

    for idx, data in enumerate(trainloader):
        print(idx, data[0].shape, data[2].shape, data[4].shape)

    # train_it = iter(trainloader)
    # l = train_it.__next__()

    # testset = ScannetDataset(root='../data/scannet_v2/scannet_pickles', npoints=8192, split='eval')
    # testset = ScannetDatasetWholeScene(root='../data/scannet_v2/scannet_pickles', npoints=8192, split='eval')
    # testset = ScannetDatasetWholeScene_evaluation(root='../data/scannet_v2/scannet_pickles', scene_list_dir='../data/scannet_v2/metadata',split='eval',block_points=8192, with_rgb=True, with_norm=True)
    # testloader = DataLoader(testset, batch_size=16, shuffle=False)

    # test_it = iter(testloader)

    # for i,j in enumerate(testloader):
        # print(i)

    # for _ in range(1000):
        # print(_)
        # l_test = test_it.__next__()

