import os
import sys
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from det3d import torchie
from det3d.torchie.apis.train import example_to_device
from .base_selector import BaseSelector
from .registry import SELECTORS
import json

# run tools/ppal_pred_list.py & tools/ppal_unc.py to get diff_category_average.json first
@SELECTORS.register_module
class PPALSelector(BaseSelector):
    def __init__(
            self,
            budget: int,
            buffer_file: str,
            dump_file_name: Optional[str] = None,
            infos_origin: List[Dict] = [],
            ent_path:  str = '/home/st2000/data/buffers/ppal_ent.pt',
            feat_path: str = '/home/st2000/data/buffers/ppal_feat.pt',
            distance_store_file: str = "/home/st2000/data/buffers/ppal_distance_map.npy",
            class_weight_file: str='tools/diff_category_average.json',
            p: int = 2,
            detector: Optional[torch.nn.Module] = None,
            dataloader: Optional[torch.utils.data.DataLoader] = None,
            logger: Optional[logging.Logger] = None,
            pred: bool = True,
            cost_b: float = 0.04,
            cost_f: float = 0.12,
            delta : int = 4 , # budget expansion ratio
    ) -> None:
        super().__init__(
            budget,
            buffer_file,
            dump_file_name,
            infos_origin=infos_origin,
            detector=detector,
            dataloader=dataloader,
            logger=logger,
            pred=pred,
            cost_b=cost_b,
            cost_f=cost_f,
        )
        self.ent_path = ent_path
        self.feat_path = feat_path
        self.distance_store_file = distance_store_file
        self.class_weight_file = class_weight_file
        assert p in [1, 2]
        self.p = p
        self.delta = delta

    def buffer_pred(self,sampled_index_list, left_index_list,**kwargs) -> torch.Tensor:
        self.logger.info(
            f"begin predict all results of samples and save them as {self.feat_path},{self.ent_path}")
        # define the cpu device to release gpu memory
        # cpu_device = torch.device("cpu"
        cuda_device = torch.device("cuda")
        prog_bar = torchie.ProgressBar(len(self.dataloader.dataset))
        b_id = 0

        #######
        with open(self.class_weight_file,'r') as f:
            class_weight = json.load(f)
        #######

        class_to_labels = {}
        label = 0
        for class_names in self.detector.bbox_head.class_names:
            for class_name in class_names:
                class_to_labels[class_name] = label
                label += 1
        label_to_class = {label: class_name for class_name, label in class_to_labels.items()}

        ent_list = []
        instance_feats_list = []
        for i, data_batch in enumerate(self.dataloader):
            with torch.no_grad():
                # if "local_rank" in kwargs:
                #     device = torch.device(kwargs["local_rank"])
                # else:
                #     device = None
                example = example_to_device(data_batch, cuda_device, non_blocking=False)
                del data_batch
                preds, fpn_feats = self.detector(example, return_loss=False, estimate=True)
                pillar_feats = fpn_feats[-1]  # [B, C, H, W]
                # dim_feat = pillar_feats.shape[1]

                # process the features and coordinates of pillars
                pillar_feats = pillar_feats.mean(dim=-1).mean(dim=-1)#.cpu()  # [B, C]

                # detector_feat.append(pillar_feats)
                for b_i, (pred, pillar_feat) in enumerate(zip(preds, pillar_feats)):
                    scores = pred["scores"]
                    entropy = -scores * torch.log(scores) - (1.0 - scores) * torch.log(1 - scores)

                    class_pred = [label_to_class[label.item()] for label in pred['label_preds']]
                    weight_ = torch.tensor( [class_weight[key] for key in class_pred] )
                    weight_ent = entropy * weight_.cuda()
                    weight_ent = weight_ent.sum()

                    ent_list.append(weight_ent)
                    instance_feats_list.append(pillar_feat)
                    prog_bar.update()
                    b_id += 1

        entropy_pred = torch.stack(ent_list)
        feat_pred = torch.stack(instance_feats_list)
        torch.save(feat_pred, self.feat_path)
        torch.save(entropy_pred, self.ent_path)

        del self.detector
        del fpn_feats
        del pillar_feats

        return entropy_pred , feat_pred

    def get_feature_distance_map(self, feats: torch.Tensor) -> torch.Tensor:
        if os.path.exists(self.distance_store_file):
            self.logger.info(f"begin to load the distance map from {self.distance_store_file}")
            distance_map = torch.from_numpy(np.load(self.distance_store_file))
            self.logger.info(f"load the distance map from {self.distance_store_file}")
        else:
            # construct Euclidean distance map
            num_infos = len(self.infos_origin)
            self.logger.info("begin to calculate the distance map")
            distance_map = torch.zeros([num_infos, num_infos])  # [num_infos, num_infos]
            for i in tqdm(range(num_infos)):
                if self.p == 1:
                    distance_vector = torch.abs(feats - feats[i]).sum(1)  # [N]
                elif self.p == 2:
                    distance_vector = torch.sqrt((feats - feats[i]) ** 2).sum(1)  # [N]
                else:
                    raise NotImplementedError

                distance_map[i] = distance_vector
            np.save(self.distance_store_file, distance_map.cpu().numpy())
            self.logger.info(f"save the distance map as {self.distance_store_file}")

        return distance_map

    def select_samples(self, **kwargs) -> None:
        # get the left index
        origin_index_list = list(range(len(self.infos_origin)))
        sampled_index_list = self.buffer[self.get_max_key()]
        left_index_list = origin_index_list.copy()
        for x in sampled_index_list:
            left_index_list.remove(x)

        if self.pred:
            ents, feats = self.buffer_pred(sampled_index_list=sampled_index_list,left_index_list=left_index_list,**kwargs)
            torch.cuda.empty_cache()
            # save the prediction results
            self.logger.info(f"all prediction results have been saved in {self.feat_path} ,{self.ent_path}")
        else:
            # load buffer
            ents = torch.load(self.ent_path)
            feats = torch.load(self.feat_path)
            # load the prediction results
            self.logger.info(f"all prediction results have been load from {self.feat_path} ,{self.ent_path}")

        self.logger.info(f"have selected {self.get_max_key()} examples, "
                         f"need to sample {self.budget} examples from "
                         f"{len(left_index_list)} examples")

        self.logger.info(f"begin to calculate distances map")

        # construct Euclidean distance map
        distance_map = self.get_feature_distance_map(feats)

        self.logger.info("get the distance map")

        ##### initial pool based on entropy
        entropy = ents[left_index_list]  # .cpu().numpy()

        sorted_preds_index = torch.argsort(-entropy).cpu().numpy().tolist()
        selected_index_list_ent = [left_index_list[sorted_preds_index[0]]]
        cost_amount = self.get_cost_amount()
        cost_amount += self.cost_f
        cost_amount += self.infos_origin[sorted_preds_index[0]]["gt_names"].shape[0] * self.cost_b
        sort_id = 1
        cost_amount_ = int(self.current_budget) + self.budget * (self.delta -1 )
        while True:
            selected_index = left_index_list[sorted_preds_index[sort_id]]
            sort_id += 1
            assert selected_index not in selected_index_list_ent, f"id: {selected_index} has been selected"

            cost_amount += self.cost_f
            cost_amount += self.infos_origin[selected_index]["gt_names"].shape[0] * self.cost_b
            if cost_amount > int(self.current_budget) + self.budget * (self.delta -1 ):
                break
            sys.stdout.write(f"cost amount: ({cost_amount}/{cost_amount_})\r")
            sys.stdout.flush()
            selected_index_list_ent.append(selected_index)

        initial_pool = selected_index_list_ent
        #####

        sampled_index_list = self.buffer[self.get_max_key()]

        # calculate the distance between sampled and unsampled
        idx_to_cal = initial_pool + sampled_index_list
        distance_map[~np.isin(np.arange(len(distance_map)), idx_to_cal)] = -np.inf
        distance_map[:, ~np.isin(np.arange(len(distance_map)), idx_to_cal)] = -np.inf
        distances = []
        for i in tqdm(sampled_index_list):
            distances.append(distance_map[i])
        distances = torch.stack(distances)  # [num_sampled, num_all]
        fps_distances = distances.min(0)[0]  # [num_all]
        selected_index_list = [int(torch.argmax(fps_distances))]

        # sample the left index and save
        cost_amount = self.get_cost_amount()
        cost_amount += self.cost_f
        cost_amount += self.infos_origin[selected_index_list[-1]]["gt_names"].shape[0] * self.cost_b
        while True:
            idx = selected_index_list[-1]
            sampled_distances = distance_map[idx]
            # update the fps distances
            fps_distances = torch.stack([fps_distances, sampled_distances])  # [2, num_left]
            fps_distances = fps_distances.min(0)[0]  # [num_left]
            selected_index = int(torch.argmax(fps_distances))
            assert selected_index not in selected_index_list, f"id: {selected_index} has been selected"

            cost_amount += self.cost_f
            cost_amount += self.infos_origin[selected_index]["gt_names"].shape[0] * self.cost_b
            if cost_amount > int(self.current_budget):
                break
            sys.stdout.write(f"cost amount: ({cost_amount}/{self.current_budget})\r")
            sys.stdout.flush()
            selected_index_list.append(selected_index)

        # get the all selected sample ids
        self.selected_index[self.current_budget] = selected_index_list + sampled_index_list