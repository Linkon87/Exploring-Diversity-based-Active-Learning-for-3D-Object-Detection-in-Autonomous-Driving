#### Download data and organise as follows

```
# For nuScenes Dataset         
└── NUSCENES_TRAINVAL_DATASET_ROOT
       ├── samples       <-- key frames
       ├── sweeps        <-- frames without annotation
       ├── maps          <-- unused
       └── v1.0-trainval <-- metadata and annotations       
```
#### Create data

Data creation should be under the gpu environment.

```
# nuScenes
python tools/create_data.py nuscenes_data_prep --root_path=$NUSCENES_TRAINVAL_DATASET_ROOT --suffix None --version="v1.0-trainval"
```

### Modify Configs

#### Update dataset setting and path

```python
dataset_type = "NuScenesDataset"
n_sweeps = 10
data_root = "/data/Datasets/nuScenes"
db_info_path="/data/Datasets/nuScenes/dbinfos_train.pkl"
train_anno = "/data/Datasets/nuScenes/infos_train_10sweeps_withvelo.pkl"
val_anno = "/data/Datasets/nuScenes/infos_val_10sweeps_withvelo.pkl"

```

## For Active Learning:

​    prepare: modify following path to your own in config_file(such as cbgs_spatial_temporal.py):

[data_root](examples/active/cbgs_spatial_temporal.py#L214)

[db_info_path](examples/active/cbgs_spatial_temporal.py#L219)

[train_anno ; val_anno](examples/active/cbgs_spatial_temporal.py#L304-L305)

[buffer_file ; log_file](examples/active/cbgs_spatial_temporal.py#L364-L366)

#### Specify Task and Anchor

**The order of tasks and anchors must be the same**

```python
tasks = [
    dict(num_class=1, class_names=["car"]),
    dict(num_class=2, class_names=["truck", "construction_vehicle"]),
    dict(num_class=2, class_names=["bus", "trailer"]),
    dict(num_class=1, class_names=["barrier"]),
    dict(num_class=2, class_names=["motorcycle", "bicycle"]),
    dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
]
anchor_generators=[
    dict(
        type="anchor_generator_range",
        sizes=[1.97, 4.63, 1.74],
        anchor_ranges=[-50.4, -50.4, -0.95, 50.4, 50.4, -0.95],
        rotations=[0, 1.57],
        velocities=[0, 0],
        matched_threshold=0.6,
        unmatched_threshold=0.45,
        class_name="car",
    ),
    dict(
        type="anchor_generator_range",
        sizes=[2.51, 6.93, 2.84],
        anchor_ranges=[-50.4, -50.4, -0.40, 50.4, 50.4, -0.40],
        rotations=[0, 1.57],
        velocities=[0, 0],
        matched_threshold=0.55,
        unmatched_threshold=0.4,
        class_name="truck",
    ),
    ...
]
```
