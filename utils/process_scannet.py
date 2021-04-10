import torch
import os
import os.path as osp
import json
import numpy as np
import csv
import pickle
from plyfile import PlyData, PlyElement

def read_aggregation(filename):
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data["segGroups"])
        for i in range(num_objects):
            object_id = data["segGroups"][i]["objectId"] + 1  # instance ids should be 1-indexed
            label = data["segGroups"][i]["label"]
            segs = data["segGroups"][i]["segments"]
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs

def read_segmentation(filename):
    assert os.path.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data["segIndices"])
        for i in range(num_verts):
            seg_id = data["segIndices"][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts

def represents_int(s):
    """ if string s represents an int. """
    try:
        int(s)
        return True
    except ValueError:
        return False

def read_label_mapping(filename, label_from="raw_category", label_to="nyu40id"):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k): v for k, v in mapping.items()}
    return mapping

test_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
def gen_label_map():
    label_map = np.zeros(41)
    for i in range(41):
        if i in test_class:
            label_map[i] = test_class.index(i)
        else:
            label_map[i] = 0
    return label_map

def reload_ply(scan_name):
    #     print(scene_id[i])
    data_file = osp.join(scannet_dir, scan_name, scan_name + \
                      "_vh_clean_2.ply")
    label_file = osp.join(scannet_dir, scan_name, scan_name + \
                      "_vh_clean_2.labels.ply")
    print(label_file)
    ply_data_tmp = PlyData.read(data_file)['vertex']
    ply_data = np.stack([
        ply_data_tmp['x'],
        ply_data_tmp['y'],
        ply_data_tmp['z'],
        ply_data_tmp['red'],
        ply_data_tmp['green'],
        ply_data_tmp['blue'],
    ],
        axis=-1
    ).astype(np.float32)
    ply_label = PlyData.read(label_file)['vertex']['label']

    keep_idx = np.where((ply_label > 0) & (ply_label < 41))
    print('original legth: {}'.format(len(keep_idx[0])))

    ply_label[np.where(ply_label == 50)] = 39
    ply_label[np.where(ply_label == 149)] = 40

    keep_idx = np.where((ply_label > 0) & (ply_label < 41))
    print('New legth: {}'.format(len(keep_idx[0])))

    new_data = ply_data[keep_idx]
    new_label = ply_label[keep_idx]

    print(new_data.shape, new_label.shape)
    return new_data, new_label

split='train'
# split='eval'
file_list = '../data/scannet_v2/metadata/scannetv2_{}.txt'.format(split)

'''loading the pickle file'''
data_root = '../data/scannet_v2'
pickle_files = osp.join(data_root, 'scannet_pickles')

with open(osp.join(pickle_files,'scannet_{}_rgb21c_pointid.pickle').format(split), 'rb') as f:
    pickle_data = pickle.load(f)
    pickle_label = pickle.load(f)
    pickle_ids = pickle.load(f)
    pickle_num_points = pickle.load(f)

with open(file_list) as fl:
    scene_names = fl.read().splitlines()

# meta_file = osp.join(
            # scannet_dir, scan_name, scan_name + ".txt"
        # )

label_map_file = '../data/scannet_v2/scannetv2-labels.combined.tsv'
# reading the label mapping from the 'floor'(607 class) to '2'(40 classes)
label_map = read_label_mapping(label_map_file)
# print('labels types', len(label_map.keys()))

def gen_instance_lables(label_map, agg_file, seg_file):

    # print(meta_file)
    
    # sth. about aligning the axis, didnt see in the pointnet processing
    # lines = open(meta_file).readlines()
    # for line in lines:
    #     if "axisAlignment" in line:
    #         print(line)
    seg_to_verts, num_verts = read_segmentation(seg_file)
    # print('num_verts',num_verts) # all the points within this scene, 2944xx

    total_num = 0
    for k in seg_to_verts.keys(): # 1856
        cur_num = len(seg_to_verts[k])
        total_num += cur_num
    # print('total_num',total_num)

    label_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
    ### ---- Instance & Label -----
    object_id_to_segs, label_to_segs = read_aggregation(agg_file)
    # label -> segs: an intermediate representation between label & vertices(points)
    # then -> seg to vertex, generate the label mapping which is [num_points] length
    for label, segs in label_to_segs.items():
        label_id = label_map[label]
    #     print(label, segs)
        for seg in segs:
            verts = seg_to_verts[seg]
            label_ids[verts] = label_id
    # print(len(label_ids)) 295486

    # A dict, len is 50, denoting 50 objs
    # 1st obj contains 112 points
    # len(object_id_to_segs[1])

    # len: 17
    # (label_to_segs)
    num_instances = len(np.unique(list(object_id_to_segs.keys()))) # num_instances
    # instance labels:
    # list of [npoints] conatininig elements of the [0(no_obj)-num_objs]
    instance_ids = np.zeros(shape=(num_verts), dtype=np.int32)# 0: unannotated
    object_id_to_label_id = {}
    for object_id, segs in object_id_to_segs.items():
    #     print(segs)
        for seg in segs:
            verts = seg_to_verts[seg]
            instance_ids[verts] = object_id
            if object_id not in object_id_to_label_id:
                object_id_to_label_id[object_id] = label_ids[verts][0]

    # Drop the unannoted points
    keep_idx = np.where((label_ids > 0) & (label_ids < 41))

    label_ids = label_ids[keep_idx]
    final_label_map = gen_label_map()
    label_ids = final_label_map[label_ids]
    instance_ids = instance_ids[keep_idx]

    return instance_ids

# DEBUG: check whether the label are the same
# print(label_ids[-100:], pickle_label[-100:])


all_instance_ids = []
for i_scan, scan_name in enumerate(scene_names):
    print('Processing {}'.format(scan_name))
    scannet_dir = '../data/scannet_v2/scans'
    # scan_name = 'scene0568_01'

    agg_file = osp.join(scannet_dir, scan_name, scan_name + ".aggregation.json")
    seg_file = osp.join(scannet_dir, scan_name, scan_name + "_vh_clean_2.0.010000.segs.json")

    cur_instance_ids = gen_instance_lables(label_map, agg_file, seg_file)
    all_instance_ids.append(cur_instance_ids)

save_dict = {}
save_dict['data'] = pickle_data
save_dict['label'] = pickle_label
for i in range(len(all_instance_ids)):
    all_instance_ids[i] = all_instance_ids[i].astype(int)
save_dict['instance'] = all_instance_ids
save_dict['npoints'] = pickle_num_points

rectify_ids = [585, 587, 823]
# rectify_ids = [823]
for ids in rectify_ids:
    new_data, new_label = reload_ply(scene_names[ids])
    save_dict['data'][ids] = new_data
    new_label_map = gen_label_map()
    new_label = new_label_map[new_label]
    save_dict['label'][ids] = new_label

for i in range(len(save_dict['data'])):
    if save_dict['data'][0].shape[0] != save_dict['instance'][0].shape[0]:
        print('Hit!')

# np.save(osp.join(pickle_files,'new_{}.npy'.format(split)), save_dict)
torch.save(save_dict, osp.join(pickle_files,'new_{}.pth'.format(split)))


