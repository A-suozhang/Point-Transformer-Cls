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


def create_input_batch(batch, is_minknet, device="cuda", quantization_size=0.05):
    if is_minknet:
        batch["coordinates"][:, 1:] = batch["coordinates"][:, 1:] / quantization_size
        return ME.TensorField(
            coordinates=batch["coordinates"], features=batch["features"], device=device,
        )
    else:
        return batch["coordinates"].permute(0, 2, 1).to(device)

def minkowski_collate_fn(list_data):
    coordinates_batch, features_batch, labels_batch = ME.utils.sparse_collate(
        [d["coordinates"] for d in list_data],
        [d["features"] for d in list_data],
        [d["label"] for d in list_data],
        dtype=torch.float32,
    )
    return {
        "coordinates": coordinates_batch,
        "features": features_batch,
        "labels": labels_batch,
    }


def stack_collate_fn(list_data):
    coordinates_batch, features_batch, labels_batch = (
        torch.stack([d["coordinates"] for d in list_data]),
        torch.stack([d["features"] for d in list_data]),
        torch.cat([d["label"] for d in list_data]),
    )

    return {
        "coordinates": coordinates_batch,
        "features": features_batch,
        "labels": labels_batch,
    }

class CoordinateTransformation:
    def __init__(self, scale_range=(0.9, 1.1), trans=0.25, jitter=0.025, clip=0.05):
        self.scale_range = scale_range
        self.trans = trans
        self.jitter = jitter
        self.clip = clip

    def __call__(self, coords):
        if random.random() < 0.9:
            coords *= np.random.uniform(
                low=self.scale_range[0], high=self.scale_range[1], size=[1, 3]
            )
        if random.random() < 0.9:
            coords += np.random.uniform(low=-self.trans, high=self.trans, size=[1, 3])
        if random.random() < 0.7:
            coords += np.clip(
                self.jitter * (np.random.rand(len(coords), 3) - 0.5),
                -self.clip,
                self.clip,
            )
        return coords

    def __repr__(self):
        return f"Transformation(scale={self.scale_range}, translation={self.trans}, jitter={self.jitter})"

def download_modelnet40_dataset(data_root):
    if not os.path.exists(data_root):
        import ipdb; ipdb.set_trace()
        raise AssertionError("No Dataset is found")
        print("Downloading the 2k downsampled ModelNet40 dataset...")
        subprocess.run(
            [
                "wget",
                "--no-check-certificate",
                "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip",
            ]
        )
        subprocess.run(["unzip", "modelnet40_ply_hdf5_2048.zip"])


class ModelNet40H5(Dataset):
    def __init__(
        self,
        phase: str,
        data_root: str = "modelnet40h5",
        transform=None,
        num_points=2048,
    ):
        Dataset.__init__(self)
        download_modelnet40_dataset(data_root)
        phase = "test" if phase in ["val", "test"] else "train"
        self.data, self.label = self.load_data(data_root, phase)
        self.transform = transform
        self.phase = phase
        self.num_points = num_points

    def load_data(self, data_root, phase):
        data, labels = [], []
        assert os.path.exists(data_root), f"{data_root} does not exist"
        files = glob.glob(os.path.join(data_root, "ply_data_%s*.h5" % phase))
        assert len(files) > 0, "No files found"
        for h5_name in files:
            with h5py.File(h5_name) as f:
                data.extend(f["data"][:].astype("float32"))
                labels.extend(f["label"][:].astype("int64"))
        data = np.stack(data, axis=0)
        labels = np.stack(labels, axis=0)
        return data, labels

    def __getitem__(self, i: int) -> dict:
        xyz = self.data[i]
        if self.phase == "train":
            np.random.shuffle(xyz)
        if len(xyz) > self.num_points:
            xyz = xyz[: self.num_points]
        if self.transform is not None:
            xyz = self.transform(xyz)
        label = self.label[i]
        xyz = torch.from_numpy(xyz)
        label = torch.from_numpy(label)
        return {
            "coordinates": xyz.to(torch.float32),
            "features": xyz.to(torch.float32),
            "label": label,
        }

    def __len__(self):
        return self.data.shape[0]

    def __repr__(self):
        return f"ModelNet40H5(phase={self.phase}, length={len(self)}, transform={self.transform})"


if __name__ == "__main__":
    dataset = ModelNet40H5(phase="train", data_root="../../data/modelnet40_ply_hdf5_2048")
    # Use stack_collate_fn for pointnet
    pointnet_data_loader = DataLoader(
        dataset, num_workers=4, collate_fn=stack_collate_fn, batch_size=16,
    )

    # Use minkowski_collate_fn for pointnet
    minknet_data_loader = DataLoader(
        dataset, num_workers=4, collate_fn=minkowski_collate_fn, batch_size=16,
    )
    import ipdb; ipdb.set_trace()

    for i, (pointnet_batch, minknet_batch) in enumerate(
        zip(pointnet_data_loader, minknet_data_loader)
    ):
        # PointNet.
        # WARNING: PointNet inputs must have the same number of points.
        pointnet_input = pointnet_batch["coordinates"].permute(0, 2, 1)

        minknet_input = ME.TensorField(
            coordinates=minknet_batch["coordinates"], features=minknet_batch["features"]
        )

