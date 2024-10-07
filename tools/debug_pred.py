import argparse
import os

import numpy as np
import torch
from det3d import __version__, torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.torchie import Config
from det3d.torchie.apis import (
    batch_processor,
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)
from det3d.torchie.trainer.utils import all_gather, synchronize

import pickle
import json

from classwise_weight.nus import NuScenesDataset_

def parse_args():
    parser = argparse.ArgumentParser(description="Test a detector")
    parser.add_argument("--config", help="train config file path",
                        default="/home/linjp/share/ActiveLearn4Detection-main/examples/active/cbgs_ppal.py",
                        )
    parser.add_argument("--work_dir", help="the dir to save logs and models",
                        default="/home/st2000/data/work_dir/seed42/badge/NUSC_CBGS_badge-debug"
                        )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )

    # parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()
    # if "LOCAL_RANK" not in os.environ:
    #     os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main():

    torch.multiprocessing.set_sharing_strategy('file_system')

    args = parse_args()
    cfg = Config.fromfile(args.config)
    # cfg.local_rank = args.local_rank

    # distributed = False

    cfg.gpus = args.gpus

    cfg.data.val.info_path = cfg.selector.infos_origin
    cfg.data.val['type']='NuScenesDataset_'
    dataset = build_dataset(cfg.data.val)

    # data_loader = build_dataloader(
    #     nuscenes_dataset,
    #     batch_size=1,#cfg.data.samples_per_gpu,
    #     workers_per_gpu=cfg.data.workers_per_gpu,
    #     num_gpus=1,
    #     dist=distributed,
    #     shuffle=False,
    # )


    # synchronize()

    # 打开保存的文件并加载数据
    with open('pred_list.pkl', 'rb') as file:
        loaded_list = pickle.load(file)
    with open('detector_feat_list.pkl', 'rb') as file:
        detector_feat_list = pickle.load(file)

    # if args.local_rank != 0:
    #
    #     return
    with open('predictions.pkl', 'rb') as file2:##验证集用来debug的数据
        predictions = pickle.load(file2)

    pred_new = {}
    for i in range(len(loaded_list)):
        for j in range(len(loaded_list[i])):
            token = loaded_list[i][j]['metadata']['token']
            pred_new[token] = loaded_list[i][j]
    pred_keys = list(pred_new.keys())
    # gt_label_list = [annotation['name'] for annotation in dataset.ground_truth_annotations]

    with open('/home/st2000/data/buffers/ppal.json', 'r') as f:
        selected_index = json.load(f)
    keys = [int(key) for key in selected_index.keys()]
    selected_index= selected_index[str(max(keys))]
    selected_index_token=[pred_keys[idx] for idx in selected_index]

    with open('dict_p_iou.pkl', 'rb') as file:
        dict_p_iou_ = pickle.load(file)
    dict_p_iou_selected = [dict_p_iou_[token_] for token_ in selected_index_token if token_ in dict_p_iou_]

    for indx in range(len(dict_p_iou_selected)):
        dict_p_iou_selected[indx].update({'quality': []})

    for i in range(0,len(dict_p_iou_selected)):
        for j in range(0,len(dict_p_iou_selected[i]['name'])):
            quality = (dict_p_iou_selected[i]['detection_score'][j]**0.6) * (dict_p_iou_selected[i]['iou'][j]**0.4)
            dict_p_iou_selected[i]['quality'].append(quality)
    # 创建一个空字典用于跟踪每个类别的总和和计数
    category_sum = {}
    category_count = {}
    # 遍历 dict_p_iou_selected
    for i in range(len(dict_p_iou_selected)):
        for j in range(len(dict_p_iou_selected[i]['name'])):
            category = dict_p_iou_selected[i]['name'][j]
            quality = dict_p_iou_selected[i]['quality'][j]
            # 更新总和和计数
            if category in category_sum:
                category_sum[category] += quality
                category_count[category] += 1
            else:
                category_sum[category] = quality
                category_count[category] = 1
    # 计算每个类别的平均值
    diff_category_average = {}
    class_weight_alpha=3.
    class_weight_ub=2.
    b = np.exp(1. / class_weight_alpha) - 1
    for category in category_sum:
        reverse_q = 1.- category_sum[category] / category_count[category]
        diff_category_average[category] = 1 + class_weight_alpha * np.log(b * reverse_q + 1) * class_weight_ub

    with open('diff_category_average.json','w') as f:
        json.dump(diff_category_average,f)

    result_dict = dataset.evaluation(pred_new, output_dir=args.work_dir)





if __name__ == "__main__":
    main()
