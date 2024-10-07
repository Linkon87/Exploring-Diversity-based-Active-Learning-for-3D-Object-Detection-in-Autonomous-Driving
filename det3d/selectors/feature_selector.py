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


@SELECTORS.register_module
class FeatureSelector(BaseSelector):
    def __init__(
            self,
            budget: int,
            buffer_file: str,
            dump_file_name: Optional[str] = None,
            infos_origin: List[Dict] = [],
            buffer_path: str = "/home/st2000/data/buffers/feature_pred.pt",
            p: int = 2,
            detector: Optional[torch.nn.Module] = None,
            dataloader: Optional[torch.utils.data.DataLoader] = None,
            logger: Optional[logging.Logger] = None,
            distance_store_file: str = "/home/st2000/data/buffers/feature_distance_map.npy",
            pred: bool = True,
            cost_b: float = 0.04,
            cost_f: float = 0.12,
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
        self.buffer_path = buffer_path
        self.distance_store_file = distance_store_file
        assert p in [1, 2]
        self.p = p

    def buffer_pred(self, **kwargs) -> torch.Tensor:
        self.logger.info(
            f"begin predict all results of samples and save them as {self.buffer_path}")
        # define the cpu device to release gpu memory
        cpu_device = torch.device("cpu")
        prog_bar = torchie.ProgressBar(len(self.dataloader.dataset))
        b_id = 0
        instance_feats_list = []
        for i, data_batch in enumerate(self.dataloader):
            with torch.no_grad():
                if "local_rank" in kwargs:
                    device = torch.device(kwargs["local_rank"])
                else:
                    device = None
                example = example_to_device(data_batch, device, non_blocking=False)
                del data_batch
                preds, fpn_feats = self.detector(example, return_loss=False, estimate=True)
                pillar_feats = fpn_feats[-1]  # [B, C, H, W]
                dim_feat = pillar_feats.shape[1]
                # process the features and coordinates of pillars
                pillar_feats = pillar_feats.mean(dim=-1).mean(dim=-1).cpu()  # [B, C]

                for b_i, pillar_feat in enumerate(pillar_feats):
                    prog_bar.update()
                    b_id += 1
                    instance_feats_list.append(pillar_feat)

        prediction = torch.stack(instance_feats_list)
        torch.save(prediction, self.buffer_path)

        del self.detector
        del fpn_feats
        del pillar_feats

        return prediction

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
            feats = self.buffer_pred(**kwargs)
            torch.cuda.empty_cache()
            # save the prediction results
            self.logger.info(f"all prediction results have been saved in {self.buffer_path}")
        else:
            # load buffer
            feats = torch.load(self.buffer_path)
            # load the prediction results
            self.logger.info(f"all prediction results have been load from {self.buffer_path}")

        self.logger.info(f"have selected {self.get_max_key()} examples, "
                         f"need to sample {self.budget} examples from "
                         f"{len(left_index_list)} examples")

        self.logger.info(f"begin to calculate distances map")

        # construct Euclidean distance map
        distance_map = self.get_feature_distance_map(feats)

        self.logger.info("get the distance map")
        sampled_index_list = self.buffer[self.get_max_key()]

        # calculate the distance between sampled and unsampled
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
