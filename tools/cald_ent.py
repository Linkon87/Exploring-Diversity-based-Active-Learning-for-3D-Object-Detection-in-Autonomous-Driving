import argparse
import os
import numpy as np
import torch
from det3d import __version__, torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.torchie import Config
import pickle
import json


from classwise_weight_cald.nus import NuScenesDataset_
import scipy.stats
from collections import defaultdict
import torch.nn as nn

def parse_args():
    parser = argparse.ArgumentParser(description="Test a detector")
    parser.add_argument("--config", help="train config file path",
                        default="examples/active/cbgs_cald.py",
                        )
    parser.add_argument("--work_dir", help="the dir to save logs and models",
                        default="/home/st2000/data/work_dir/seed42/cald"
                        )
    parser.add_argument("--selected_buffer",
                        default="/home/st2000/data/buffers/cald.json"
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


    with open('pred_list-aug.pkl', 'rb') as file:
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

    for key in pred_new.keys():
        pred_new[key].update({'consistency_img': 1.0})

    for key in dict_p_iou_.keys():
        dict_p_iou_[key].update({'consistency_img':1.0})

    for key in dict_p_iou_.keys():
        for i in range(0,len(dict_p_iou_[key]['name'])):
            q = dict_p_iou_[key]['detection_score'][i]
            p = dict_p_iou_[key]['ref_score'][i]
            m = (p + q) / 2
            js = 0.5 * scipy.stats.entropy(p, m) + 0.5 * scipy.stats.entropy(q, m)
            if js < 0:
                js = 0
            dict_p_iou_[key]['consistency_img'] = min(dict_p_iou_[key]['consistency_img'] , np.abs(
                dict_p_iou_[key]['iou'][i]   + 0.5 * (1 - js) * (p+q) - 1.3).item())
            pred_new[key].update({'consistency_img':dict_p_iou_[key]['consistency_img']})

    sorted_indices = [index for index, _ in
                      sorted(enumerate(pred_new.keys()), key=lambda x: pred_new[x[1]]['consistency_img'])]

    with open('/home/st2000/data/buffers/cald_ent_sorted_idx.json', 'w') as f:
        json.dump(sorted_indices, f)

    dict_p_iou_selected = [dict_p_iou_[token_] for token_ in selected_index_token if token_ in dict_p_iou_]

    sorted_token_list = sorted(pred_new.keys(), key=lambda x: pred_new[x]['consistency_img'])
    sorted_idx_and_token = {}
    for i in range(len(sorted_indices)):
        key = sorted_indices[i]
        value = sorted_token_list[i]
        sorted_idx_and_token[value] = key

    sorted_res_ = {}
    idx_list = []
    for tk in sorted_token_list:
        if tk in dict_p_iou_.keys():
            sorted_res_.update({tk:dict_p_iou_[tk]})
            idx_list.append(sorted_idx_and_token[tk])

    category_sum = {}
    category_count = {}
    for i in range(len(dict_p_iou_selected)):
        for j in range(len(dict_p_iou_selected[i]['name'])):
            category = dict_p_iou_selected[i]['name'][j]

            if category in category_sum:
                category_sum[category] += 0
                category_count[category] += 1
            else:
                category_sum[category] = 0
                category_count[category] = 1

    name_to_class = {value: key for key, value in enumerate(category_sum.keys())}

    for key in sorted_res_.keys():
        for i in range(len(sorted_res_[key]['name'])):
            name = sorted_res_[key]['name'][i]
            sorted_res_[key]['name'][i] = name_to_class[name]

    img_cls = []
    for key in sorted_res_.keys():
        cls_corr = [0] * 10
        for i in range(len(sorted_res_[key]['name'])):
            cls_corr[sorted_res_[key]['name'][i]] += 1
        img_cls.append(cls_corr)

    category_count_new = {}
    for old_key,value in category_count.items():
        new_key = name_to_class[old_key]
        category_count_new[new_key] = value

    num_labeled_samples = len(selected_index)
    num_classes = 10

    with torch.no_grad():
        result = []
        for i in range(num_labeled_samples):
            cls_corr = [0] * num_classes
            for cls, count in category_count_new.items():
                cls_corr[cls] = count
            result.append(cls_corr)
            _result = torch.tensor(np.mean(np.array(result), axis=0)).unsqueeze(0) #labeled distribution
        p = torch.nn.functional.softmax(_result, -1)
        q = torch.nn.functional.softmax(torch.tensor((img_cls),dtype=torch.float64), -1)
        log_mean = ((p + q) / 2).log()
        KLDivLoss = nn.KLDivLoss(reduction='none')
        jsdiv = torch.sum(KLDivLoss(log_mean, p), dim=1) / 2 + torch.sum(KLDivLoss(log_mean, q), dim=1) / 2

        idx_to_jsdiv = {}
        for i in range(len(idx_list)):
            idx_to_jsdiv[idx_list[i]] = jsdiv[i].item()

        with open('idx_to_jsdiv.pkl', 'wb') as file:
                pickle.dump(idx_to_jsdiv,file)





if __name__ == "__main__":
    main()
