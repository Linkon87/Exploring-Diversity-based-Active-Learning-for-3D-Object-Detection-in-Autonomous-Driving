from .kitti import KittiDataset
from .nuscenes import NuScenesDataset, NuScenesPartialDataset
from .lyft import LyftDataset

dataset_factory = {
    "KITTI": KittiDataset,
    "NUSC": NuScenesDataset,
    "NUSC_PART": NuScenesPartialDataset,
    "LYFT": LyftDataset,
}


def get_dataset(dataset_name):
    return dataset_factory[dataset_name]
