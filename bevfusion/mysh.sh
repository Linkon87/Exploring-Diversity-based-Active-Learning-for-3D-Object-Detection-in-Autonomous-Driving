#!/bin/bash

python tools/create_data.py nuscenes \
    --root-path /dataset --out-dir /dataset \
    --extra-tag nuscenes --budget 4800 \
    --buffer_path /dataset/buffers/seed42/uwe-seed42.json

torchpack dist-run -np 4 python tools/train.py \
 configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml \
 --run-dir /dataset/runs/uwe4800_seed42/lidar-only --optimizer.lr 0.0000646 --data.samples_per_gpu 6


torchpack dist-run -np 4 python tools/train.py \
 configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
 --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
 --load_from /dataset/runs/uwe4800_seed42/lidar-only/epoch_20.pth \
 --run-dir /dataset/runs/uwe4800_seed42/fus

torchpack dist-run -np 4 python tools/test.py \
 configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  /dataset/runs/uwe4800_seed42/fus/latest.pth \
  --eval bbox --eval-options "jsonfile_prefix=/dataset/runs/uwe4800_seed42/fus/val/"  