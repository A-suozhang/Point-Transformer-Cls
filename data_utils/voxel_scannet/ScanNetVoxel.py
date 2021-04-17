# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.
import glob
from abc import ABC
from tqdm import tqdm
import os.path as osp
from pathlib import Path
from collections import defaultdict
import random
import numpy as np
from enum import Enum
import logging

import torch


from torch.utils.data import Dataset, DataLoader

# from lib.pc_utils import read_plyfile, save_point_cloud
import transforms as t
from sparse_voxelization import SparseVoxelizer
# from lib.dataloader import InfSampler
import MinkowskiEngine as ME


'''
the ply_reader
'''
from plyfile import PlyData, PlyElement

def read_plyfile(filepath):
  """Read ply file and return it as numpy array. Returns None if emtpy."""
  with open(filepath, 'rb') as f:
    plydata = PlyData.read(f)
  if plydata.elements:
    return pd.DataFrame(plydata.elements[0].data).values

# Maybe not used at all?
def save_point_cloud(points_3d, filename, binary=True, with_label=False, verbose=True):
  """Save an RGB point cloud as a PLY file.

  Args:
    points_3d: Nx6 matrix where points_3d[:, :3] are the XYZ coordinates and points_3d[:, 4:] are
        the RGB values. If Nx3 matrix, save all points with [128, 128, 128] (gray) color.
  """
  assert points_3d.ndim == 2
  if with_label:
    assert points_3d.shape[1] == 7
    python_types = (float, float, float, int, int, int, int)
    npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                 ('blue', 'u1'), ('label', 'u1')]
  else:
    if points_3d.shape[1] == 3:
      gray_concat = np.tile(np.array([128], dtype=np.uint8), (points_3d.shape[0], 3))
      points_3d = np.hstack((points_3d, gray_concat))
    assert points_3d.shape[1] == 6
    python_types = (float, float, float, int, int, int)
    npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                 ('blue', 'u1')]
  if binary is True:
    # Format into NumPy structured array
    vertices = []
    for row_idx in range(points_3d.shape[0]):
      cur_point = points_3d[row_idx]
      vertices.append(tuple(dtype(point) for dtype, point in zip(python_types, cur_point)))
    vertices_array = np.array(vertices, dtype=npy_types)
    el = PlyElement.describe(vertices_array, 'vertex')

    # Write
    PlyData([el]).write(filename)
  else:
    # PlyData([el], text=True).write(filename)
    with open(filename, 'w') as f:
      f.write('ply\n'
              'format ascii 1.0\n'
              'element vertex %d\n'
              'property float x\n'
              'property float y\n'
              'property float z\n'
              'property uchar red\n'
              'property uchar green\n'
              'property uchar blue\n'
              'property uchar alpha\n'
              'end_header\n' % points_3d.shape[0])
      for row_idx in range(points_3d.shape[0]):
        X, Y, Z, R, G, B = points_3d[row_idx]
        f.write('%f %f %f %d %d %d 0\n' % (X, Y, Z, R, G, B))
  if verbose is True:
    print('Saved point cloud to: %s' % filename)


'''
The InfDataloader
'''
from torch.utils.data.sampler import Sampler
class InfSampler(Sampler):
  """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

  def __init__(self, data_source, shuffle=False):
    self.data_source = data_source
    self.shuffle = shuffle
    self.reset_permutation()

  def reset_permutation(self):
    perm = len(self.data_source)
    if self.shuffle:
      perm = torch.randperm(perm)
    self._perm = perm.tolist()

  def __iter__(self):
    return self

  def __next__(self):
    if len(self._perm) == 0:
      self.reset_permutation()
    return self._perm.pop()

  def __len__(self):
    return len(self.data_source)

  next = __next__  # Python 2 compatibility


def read_txt(path):
  """Read txt file into lines.
  """
  with open(path) as f:
    lines = f.readlines()
  lines = [x.strip() for x in lines]
  return lines


class DatasetPhase(Enum):
  Train = 0
  Val = 1
  Val2 = 2
  TrainVal = 3
  Test = 4


def datasetphase_2str(arg):
  if arg == DatasetPhase.Train:
    return 'train'
  elif arg == DatasetPhase.Val:
    return 'val'
  elif arg == DatasetPhase.Val2:
    return 'val2'
  elif arg == DatasetPhase.TrainVal:
    return 'trainval'
  elif arg == DatasetPhase.Test:
    return 'test'
  else:
    raise ValueError('phase must be one of dataset enum.')


def str2datasetphase_type(arg):
  if arg.upper() == 'TRAIN':
    return DatasetPhase.Train
  elif arg.upper() == 'VAL':
    return DatasetPhase.Val
  elif arg.upper() == 'VAL2':
    return DatasetPhase.Val2
  elif arg.upper() == 'TRAINVAL':
    return DatasetPhase.TrainVal
  elif arg.upper() == 'TEST':
    return DatasetPhase.Test
  else:
    raise ValueError('phase must be one of train/val/test')


def cache(func):

  def wrapper(self, *args, **kwargs):
    # Assume that args[0] is index
    index = args[0]
    if self.cache:
      if index not in self.cache_dict[func.__name__]:
        results = func(self, *args, **kwargs)
        self.cache_dict[func.__name__][index] = results
      return self.cache_dict[func.__name__][index]
    else:
      return func(self, *args, **kwargs)

  return wrapper

'''
A few base DatsetClasses
'''

class DictDataset(Dataset, ABC):

  IS_CLASSIFICATION = False
  IS_ONLINE_VOXELIZATION = False
  NEED_PRED_POSTPROCESSING = False
  IS_FULL_POINTCLOUD_EVAL = False

  def __init__(self,
               data_paths,
               input_transform=None,
               target_transform=None,
               cache=False,
               data_root='/'):
    """
    data_paths: list of lists, [[str_path_to_input, str_path_to_label], [...]]
    """
    Dataset.__init__(self)

    # Allows easier path concatenation
    if not isinstance(data_root, Path):
      data_root = Path(data_root)

    self.data_root = data_root
    self.data_paths = sorted(data_paths)
    self.input_transform = input_transform
    self.target_transform = target_transform

    # dictionary of input
    self.data_loader_dict = {
        'input': (self.load_input, self.input_transform),
        'target': (self.load_target, self.target_transform)
    }

    # For large dataset, do not cache
    self.cache = cache
    self.cache_dict = defaultdict(dict)
    self.loading_key_order = ['input', 'target']

  def load_ply(self, index, data_index=0):
    filepath = self.data_root / self.data_paths[index][data_index]
    return self.read_ply(filepath)

  def load_input(self, index):
    raise NotImplementedError

  def load_target(self, index):
    raise NotImplementedError

  def get_classnames(self):
    pass

  def reorder_result(self, result):
    return result

  def __getitem__(self, index):
    out_array = []
    for k in self.loading_key_order:
      loader, transformer = self.data_loader_dict[k]
      v = loader(index)
      if transformer:
        v = transformer(v)
      out_array.append(v)
    return out_array

  def __len__(self):
    return len(self.data_paths)


class VoxelizationDatasetBase(DictDataset, ABC):
  IS_TEMPORAL = False
  CLIP_SIZE = 1000
  CLIP_BOUND = (-1000, -1000, -1000, 1000, 1000, 1000)
  ROTATION_AXIS = None
  LOCFEAT_IDX = None
  TRANSLATION_AUG = 0.
  INPUT_SPATIAL_DIM = (128, 128, 128)
  OUTPUT_SPATIAL_DIM = (128, 128, 128)
  NUM_IN_CHANNEL = None
  NUM_LABELS = -1  # Number of labels in the dataset, including all ignore classes
  IGNORE_LABELS = None  # labels that are not evaluated
  IS_ONLINE_VOXELIZATION = True

  def __init__(self,
               data_paths,
               input_transform=None,
               target_transform=None,
               cache=False,
               data_root='/',
               explicit_rotation=-1,
               ignore_mask=255,
               return_transformation=False,
               **kwargs):
    """
    ignore_mask: label value for ignore class. It will not be used as a class in the loss or evaluation.
    explicit_rotation: # of discretization of 360 degree. # data would be num_data * explicit_rotation
    """
    DictDataset.__init__(
        self,
        data_paths,
        input_transform=input_transform,
        target_transform=target_transform,
        cache=cache,
        data_root=data_root)

    self.ignore_mask = ignore_mask
    self.explicit_rotation = explicit_rotation
    self.return_transformation = return_transformation

  def __getitem__(self, index):
    raise NotImplementedError

  def load_ply(self, index):
    assert "when loading scannet, this func should be override"
    filepath = self.data_root / self.data_paths[index]
    return read_plyfile(filepath), None

  def __len__(self):
    num_data = len(self.data_paths)
    if self.explicit_rotation > 1:
      return num_data * self.explicit_rotation
    return num_data


class SparseVoxelizationDataset(VoxelizationDatasetBase):
  """This dataset loads RGB point clouds and their labels as a list of points
  and voxelizes the pointcloud with sufficient data augmentation.
  """
  # Voxelization arguments
  CLIP_BOUND = None
  VOXEL_SIZE = 0.05  # 5cm

  # Augmentation arguments
  SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
  ROTATION_AUGMENTATION_BOUND = ((-np.pi / 6, np.pi / 6), (-np.pi, np.pi), (-np.pi / 6, np.pi / 6))
  TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.05, 0.05), (-0.2, 0.2))
  ELASTIC_DISTORT_PARAMS = None
  PREVOXELIZE_VOXEL_SIZE = None

  def __init__(self,
               data_paths,
               input_transform=None,
               target_transform=None,
               data_root='/',
               explicit_rotation=-1,
               ignore_label=255,
               return_transformation=False,
               augment_data=False,
               elastic_distortion=False,
               **kwargs):
    self.augment_data = augment_data
    self.elastic_distortion = elastic_distortion
    VoxelizationDatasetBase.__init__(
        self,
        data_paths,
        input_transform=input_transform,
        target_transform=target_transform,
        cache=cache,
        data_root=data_root,
        ignore_mask=ignore_label,
        return_transformation=return_transformation)

    self.sparse_voxelizer = SparseVoxelizer(
        voxel_size=self.VOXEL_SIZE,
        clip_bound=self.CLIP_BOUND,
        use_augmentation=augment_data,
        scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
        rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
        translation_augmentation_ratio_bound=self.TRANSLATION_AUGMENTATION_RATIO_BOUND,
        rotation_axis=self.LOCFEAT_IDX,
        ignore_label=ignore_label)

    # map labels not evaluated to ignore_label
    label_map = {}
    n_used = 0
    for l in range(self.NUM_LABELS):
      if l in self.IGNORE_LABELS:
        label_map[l] = self.ignore_mask
      else:
        label_map[l] = n_used
        n_used += 1
    label_map[self.ignore_mask] = self.ignore_mask
    self.label_map = label_map
    self.NUM_LABELS -= len(self.IGNORE_LABELS)

    '''
    Loading all point-clouds from the pth at first
    then use the inedxing in `self.load_ply`
    '''
    self.data_dict = torch.load(data_paths)

  # def get_output_id(self, iteration):
    # return self.data_paths[iteration]

  def convert_mat2cfl(self, mat):
    # Generally, xyz,rgb,label
    return mat[:, :3], mat[:, 3:-1], mat[:, -1]

  def _augment_elastic_distortion(self, pointcloud):
    if self.ELASTIC_DISTORT_PARAMS is not None:
      if random.random() < 0.95:
        for granularity, magnitude in self.ELASTIC_DISTORT_PARAMS:
          pointcloud = t.elastic_distortion(pointcloud, granularity, magnitude)
    return pointcloud

  def load_ply(self, index):
      '''
      rewrite the loading from the pth file,
      unpack it into new form
      maybe has 7 dims, or maybe instance label?
      '''
      coords = self.data_dict['data'][index][:,:3]
      features = self.data_dict['data'][index][:,3:]
      labels = self.data_dict['label'][index]

      return [coords, features, labels]

  def __getitem__(self, index):
    # default -1
    # if self.explicit_rotation > 1:
      # rotation_space = np.linspace(-np.pi, np.pi, self.explicit_rotation + 1)
      # rotation_angle = rotation_space[index % self.explicit_rotation]
      # index //= self.explicit_rotation
    # else:
      # rotation_angle = None
    rotation_angle = None

    '''as shown in the load_ply from above, no center is applied'''
    pointcloud = self.load_ply(index)

    # also none by default
    # if self.PREVOXELIZE_VOXEL_SIZE is not None:
      # inds = ME.SparseVoxelize(pointcloud[:, :3] / self.PREVOXELIZE_VOXEL_SIZE, return_index=True)
      # pointcloud = pointcloud[inds]

    '''TODO: what is distortion?, default is false, ok'''
    # if self.elastic_distortion:
      # pointcloud = self._augment_elastic_distortion(pointcloud)

    # import open3d as o3d
    # from lib.open3d_utils import make_pointcloud
    # pcd = make_pointcloud(np.floor(pointcloud[:, :3] / self.PREVOXELIZE_VOXEL_SIZE))
    # o3d.draw_geometries([pcd])

    # coords, feats, labels = self.convert_mat2cfl(pointcloud)
    center = None
    coords, feats, labels = pointcloud

    outs = self.sparse_voxelizer.voxelize(
        coords,
        feats,
        labels,
        center=center,
        rotation_angle=rotation_angle,
        return_transformation=self.return_transformation)

    if self.return_transformation:
      coords, feats, labels, transformation = outs
      transformation = np.expand_dims(transformation, 0)
    else:
      coords, feats, labels = outs

    # dont know why became tensor, maybe ME update
    coords = coords.numpy()

    # map labels not used for evaluation to ignore_label
    if self.input_transform is not None:
      coords, feats, labels = self.input_transform(coords, feats, labels)
    if self.target_transform is not None:
      coords, feats, labels = self.target_transform(coords, feats, labels)
    '''
    ignore labels is done when loadng the pth data, since when we generate the pth,we already do 0<labels<40
    the raw points(from segs.json) as example:
    for scene_id[70]: scene_0585_00,
    the vertex contains 245108 points, however after ds there are only 202540(in pth)
    '''
    # if self.IGNORE_LABELS is not None:
      # labels = np.array([self.label_map[x] for x in labels], dtype=np.int)

    return_args = [coords, feats, labels]
    if self.return_transformation:
      return_args.extend([pointcloud.astype(np.float32), transformation.astype(np.float32)])
    return tuple(return_args)

  def cleanup(self):
    self.sparse_voxelizer.cleanup()

  def __len__(self):
    return len(self.data_dict['data'])

def initialize_data_loader(DatasetClass,
                           data_root,
                           phase,
                           threads,
                           shuffle,
                           repeat,
                           augment_data,
                           batch_size,
                           limit_numpoints,
                           elastic_distortion=False,
                           input_transform=None,
                           target_transform=None):
  if isinstance(phase, str):
    phase = str2datasetphase_type(phase)

    # default is false
  # if config.return_transformation:
    # collate_fn = t.cflt_collate_fn_factory(limit_numpoints)
  # else:
    # collate_fn = t.cfl_collate_fn_factory(limit_numpoints)
  collate_fn = t.cfl_collate_fn_factory(limit_numpoints)

  input_transforms = []
  # default None
  if input_transform is not None:
    input_transforms += input_transform

  if augment_data: # TRUE
    input_transforms += [
        t.RandomDropout(0.2),
        t.RandomHorizontalFlip(DatasetClass.ROTATION_AXIS, DatasetClass.IS_TEMPORAL),
        t.ChromaticAutoContrast(),
        t.ChromaticTranslation(0.1),
        t.ChromaticJitter(0.05),
        # t.HueSaturationTranslation(config.data_aug_hue_max, config.data_aug_saturation_max),
    ]

  if len(input_transforms) > 0:
    input_transforms = t.Compose(input_transforms)
  else:
    input_transforms = None

  dataset = DatasetClass(
      data_root,
      input_transform=input_transforms,
      target_transform=target_transform,
      cache=False,
      augment_data=augment_data,
      elastic_distortion=elastic_distortion,
      phase=phase)

  if repeat:
    # Use the inf random sampler
    data_loader = DataLoader(
        dataset=dataset,
        num_workers=threads,
        batch_size=batch_size,
        collate_fn=collate_fn,
        sampler=InfSampler(dataset, shuffle))
  else:
    # Default shuffle=False
    data_loader = DataLoader(
        dataset=dataset,
        num_workers=threads,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle)

  return data_loader


def generate_meta(voxels_path,
                  split_path,
                  get_area_fn,
                  trainarea,
                  valarea,
                  testarea,
                  data_root='/',
                  check_pc=False):
  train_file_list = []
  val_file_list = []
  test_file_list = []
  for pointcloud_file in tqdm(glob.glob(osp.join(data_root, voxels_path))):
    area = get_area_fn(pointcloud_file)
    if area in trainarea:
      file_list = train_file_list
    elif area in valarea:
      file_list = val_file_list
    elif area in testarea:
      file_list = test_file_list
    else:
      raise ValueError('Area %s not in the split' % area)

    # Skip label files.
    if pointcloud_file.endswith('_label_voxel.ply'):
      continue

    # Parse and check if the corresponding label file exists.
    file_stem, file_ext = osp.splitext(pointcloud_file)
    file_stem_split = file_stem.split('_')
    file_stem_split.insert(-1, 'label')

    pointcloud_label_file = '_'.join(file_stem_split) + file_ext
    if not osp.isfile(pointcloud_label_file):
      raise ValueError('Lable file missing for: ' + pointcloud_file)

    # Check if the pointcloud is empty.
    if check_pc:
      pointcloud_data = read_plyfile(pointcloud_file)
      if not pointcloud_data:
        print('Skipping empty point cloud: %s.')
        continue

    pointcloud_file = osp.relpath(pointcloud_file, data_root)
    pointcloud_label_file = osp.relpath(pointcloud_label_file, data_root)

    # Append metadata.
    file_list.append([pointcloud_file, pointcloud_label_file])

  with open(split_path % 'train', 'w') as f:
    f.write('\n'.join([' '.join(pair) for pair in train_file_list]))
  with open(split_path % 'val', 'w') as f:
    f.write('\n'.join([' '.join(pair) for pair in val_file_list]))
  with open(split_path % 'test', 'w') as f:
    f.write('\n'.join([' '.join(pair) for pair in test_file_list]))



'''
============================ The main body of the ScannetDataLoader =====================================
'''

CLASS_LABELS = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
                'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture')
VALID_CLASS_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
TEST_FULL_PLY_PATH = 'test/%s_vh_clean_2.ply'
FULL_EVAL_PATH = 'outputs/fulleval'
SCANNET_COLOR_MAP = {
    0: (0., 0., 0.),
    1: (174., 199., 232.),
    2: (152., 223., 138.),
    3: (31., 119., 180.),
    4: (255., 187., 120.),
    5: (188., 189., 34.),
    6: (140., 86., 75.),
    7: (255., 152., 150.),
    8: (214., 39., 40.),
    9: (197., 176., 213.),
    10: (148., 103., 189.),
    11: (196., 156., 148.),
    12: (23., 190., 207.),
    14: (247., 182., 210.),
    15: (66., 188., 102.),
    16: (219., 219., 141.),
    17: (140., 57., 197.),
    18: (202., 185., 52.),
    19: (51., 176., 203.),
    20: (200., 54., 131.),
    21: (92., 193., 61.),
    22: (78., 71., 183.),
    23: (172., 114., 82.),
    24: (255., 127., 14.),
    25: (91., 163., 138.),
    26: (153., 98., 156.),
    27: (140., 153., 101.),
    28: (158., 218., 229.),
    29: (100., 125., 154.),
    30: (178., 127., 135.),
    32: (146., 111., 194.),
    33: (44., 160., 44.),
    34: (112., 128., 144.),
    35: (96., 207., 209.),
    36: (227., 119., 194.),
    37: (213., 92., 176.),
    38: (94., 106., 211.),
    39: (82., 84., 163.),
    40: (100., 85., 144.),
}


class ScannetSparseVoxelizationDataset(SparseVoxelizationDataset):

  # Voxelization arguments
  CLIP_BOUND = None
  TEST_CLIP_BOUND = None
  VOXEL_SIZE = 0.05

  # Augmentation arguments
  ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,
                                                                                        np.pi))
  TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
  ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))
  # Original SCN uses
  # ELASTIC_DISTORT_PARAMS = ((2, 4), (8, 8))

  ROTATION_AXIS = 'z'
  LOCFEAT_IDX = 2
  NUM_LABELS = 41  # Will be converted to 20 as defined in IGNORE_LABELS.
  IGNORE_LABELS = tuple(set(range(41)) - set(VALID_CLASS_IDS))
  IS_FULL_POINTCLOUD_EVAL = True

  DATA_PATH_FILE = {
      DatasetPhase.Train: 'scannetv2_train.txt',
      DatasetPhase.Val: 'scannetv2_val.txt',
      DatasetPhase.TrainVal: 'trainval_uncropped.txt',
      DatasetPhase.Test: 'scannetv2_test.txt'
  }

  def __init__(self,
               data_root,
               input_transform=None,
               target_transform=None,
               augment_data=True,
               elastic_distortion=False,
               cache=False,
               phase=DatasetPhase.Train):
    if isinstance(phase, str):
      phase = str2datasetphase_type(phase)
    # Use cropped rooms for train/val
    if phase not in [DatasetPhase.Train, DatasetPhase.TrainVal]:
      self.CLIP_BOUND = self.TEST_CLIP_BOUND
    '''rewrite the datapaths as the pth file'''
    data_paths = osp.join(data_root, 'new_{}.pth'.format(datasetphase_2str(phase).lower()))
    # data_paths = read_txt(osp.join(data_root, 'metadata',self.DATA_PATH_FILE[phase]))
    logging.info('Loading {}: {}'.format(self.__class__.__name__, self.DATA_PATH_FILE[phase]))
    super().__init__(
        data_paths,
        data_root=data_root,
        input_transform=input_transform,
        target_transform=target_transform,
        ignore_label=False,
        return_transformation=False,
        augment_data=augment_data,
        elastic_distortion=elastic_distortion)

  def get_output_id(self, iteration):
    return '_'.join(Path(self.data_paths[iteration]).stem.split('_')[:2])


class ScannetSparseVoxelization2cmDataset(ScannetSparseVoxelizationDataset):
  VOXEL_SIZE = 0.02

if __name__ == '__main__':
    # TEST the ScanNet Voxel

    # trainset = ScannetSparseVoxelizationDataset(
               # data_root='../../data/scannet_v2/scannet_pickles',
               # input_transform=None,
               # target_transform=None,
               # augment_data=True,
               # elastic_distortion=False,
               # cache=False,
               # phase=DatasetPhase.Train)
    # trainset.__getitem__(0)

    train_loader = initialize_data_loader(
        DatasetClass=ScannetSparseVoxelization2cmDataset,
        data_root='../../data/scannet_v2/scannet_pickles',
        phase="train",
        threads=4, # num-workers
        shuffle=True,
        repeat=False,
        augment_data=True,
        batch_size=16,
        limit_numpoints=1200000,
    )

    testloader = initialize_data_loader(
        DatasetClass=ScannetSparseVoxelizationDataset,
        data_root='../../data/scannet_v2/scannet_pickles',
        phase="val",
        threads=4, # num-workers
        shuffle=False,
        repeat=False,
        augment_data=False,
        batch_size=16,
        limit_numpoints=False,
    )


    # dat = iter(train_loader).__next__()
    print('done loading.')

    for idx, dat in enumerate(train_loader):
        print(dat[0]).shape
        import ipdb; ipdb.set_trace()
        print(len(dat))
