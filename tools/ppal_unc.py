import argparse
import os
import numpy as np
import torch
from det3d import __version__, torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.torchie import Config
import pickle
import json

from classwise_weight.nus import NuScenesDataset_

def parse_args():
    parser = argparse.ArgumentParser(description="Test a detector")
    parser.add_argument("--config", help="train config file path",
                        default="/home/linjp/share/ActiveLearn4Detection-main/examples/active/cbgs_ppal.py",
                        )
    parser.add_argument("--work_dir", help="the dir to save logs and models",
                        default="/home/st2000/data/work_dir/seed0/ppal"
                        )
    parser.add_argument("--selected_buffer",
                        default="/home/st2000/data/buffers/ppal.json"
                        )
    parser.add_argument(
        "--gpus",
        type=int,
        default=2,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    cfg.gpus = args.gpus

    cfg.data.val.info_path = cfg.selector.infos_origin
    cfg.data.val['type']='NuScenesDataset_'
    dataset = build_dataset(cfg.data.val)


    with open('pred_list.pkl', 'rb') as file:
        loaded_list = pickle.load(file)


    pred_new = {}
    for i in range(len(loaded_list)):
        for j in range(len(loaded_list[i])):
            token = loaded_list[i][j]['metadata']['token']
            pred_new[token] = loaded_list[i][j]
    pred_keys = list(pred_new.keys())

    with open(args.selected_buffer, 'r') as f:
        selected_index = json.load(f)
    keys = [int(key) for key in selected_index.keys()]
    selected_index= selected_index[str(max(keys))]
    selected_index_token=[pred_keys[idx] for idx in selected_index]

    if os.path.exists('dict_p_iou.pkl'):
        with open('dict_p_iou.pkl', 'rb') as file:
            dict_p_iou_ = pickle.load(file)
    else:
        dict_p_iou_ = dataset.evaluation(pred_new, output_dir=args.work_dir)

    dict_p_iou_selected = [dict_p_iou_[token_] for token_ in selected_index_token if token_ in dict_p_iou_]

    for indx in range(len(dict_p_iou_selected)):
        dict_p_iou_selected[indx].update({'quality': []})

    for i in range(0,len(dict_p_iou_selected)):
        for j in range(0,len(dict_p_iou_selected[i]['name'])):
            quality = (dict_p_iou_selected[i]['detection_score'][j]**0.6) * (dict_p_iou_selected[i]['iou'][j]**0.4)
            dict_p_iou_selected[i]['quality'].append(quality)

    category_sum = {}
    category_count = {}
    for i in range(len(dict_p_iou_selected)):
        for j in range(len(dict_p_iou_selected[i]['name'])):
            category = dict_p_iou_selected[i]['name'][j]
            quality = dict_p_iou_selected[i]['quality'][j]

            if category in category_sum:
                category_sum[category] += quality
                category_count[category] += 1
            else:
                category_sum[category] = quality
                category_count[category] = 1
    diff_category_average = {}
    class_weight_alpha=3.
    class_weight_ub=2.
    b = np.exp(1. / class_weight_alpha) - 1
    for category in category_sum:
        reverse_q = 1.- category_sum[category] / category_count[category]
        diff_category_average[category] = 1 + class_weight_alpha * np.log(b * reverse_q + 1) * class_weight_ub

    with open('diff_category_average.json','w') as f:
        json.dump(diff_category_average,f)






if __name__ == "__main__":
    main()
