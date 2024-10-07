import random
import logging
import sys
from typing import Dict, List, Optional
import torch
import numpy as np
from tqdm import tqdm
from .base_selector import BaseSelector
from .registry import SELECTORS
from det3d.torchie.fileio import load


@SELECTORS.register_module
class TemporalSelector(BaseSelector):
    def __init__(
            self,
            budget: int,
            buffer_file: str,
            dump_file_name: Optional[str] = None,
            infos_origin: List[Dict] = [],
            detector: Optional[torch.nn.Module] = None,
            dataloader: Optional[torch.utils.data.DataLoader] = None,
            logger: Optional[logging.Logger] = None,
            pred: bool = False,
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

    def select_samples(self, **kwargs) -> None:
        # NOTE: find the connection frmaes of different video
        note_frames = []
        for i, (info_prev, info_curr) in enumerate(zip(self.infos_origin[:-1], self.infos_origin[1:])):
            prev_logfile = info_prev["cam_front_path"].split("/")[-1].split("__")[0]
            curr_logfile = info_curr["cam_front_path"].split("/")[-1].split("__")[0]
            if prev_logfile != curr_logfile:
                note_frames.append(i)

        logfile2frames = {}
        for i, info in enumerate(tqdm(self.infos_origin)):
            logfile = info["cam_front_path"].split("/")[-1].split("__")[0]
            if logfile not in logfile2frames:
                logfile2frames[logfile] = []
            logfile2frames[logfile].append(i)

        # construct Euclidean distance map
        num_infos = len(self.infos_origin)
        self.logger.info("begin to calculate the distance map")
        margin = 1e6
        distance_map = np.ones([num_infos, num_infos]) * margin
        for logfile, frames in tqdm(logfile2frames.items()):
            for frame in frames:
                distance_map[frame, frames] = np.abs(np.array(frames) - frame)

        self.logger.info("get the distance map")
        sampled_index_list = self.buffer[self.get_max_key()]

        # calculate the distance between sampled and unsampled
        distances = []
        if len(sampled_index_list) > 0:
            for i in tqdm(sampled_index_list):
                distances.append(distance_map[i])
            distances = np.stack(distances)  # [num_sampled, num_all]
            fps_distances = distances.min(0)  # [num_all]
            selected_index_list = [int(np.argmax(fps_distances))]
        else:
            selected_index_list = [random.choice(range(len(self.infos_origin)))]
            fps_distances = distance_map[selected_index_list[-1]]

        # sample the left index(fps) and save
        cost_amount = self.get_cost_amount()
        cost_amount += self.cost_f
        cost_amount += self.infos_origin[selected_index_list[-1]]["gt_names"].shape[0] * self.cost_b
        while True:
            idx = selected_index_list[-1]
            sampled_distances = distance_map[idx]
            # update the fps distances
            fps_distances = np.stack([fps_distances, sampled_distances])  # [2, num_left]
            fps_distances = fps_distances.min(0)  # [num_left]
            selected_index = int(np.argmax(fps_distances))
            assert selected_index not in sampled_index_list, f"id: {selected_index} not in the left index list"
            assert selected_index not in selected_index_list, f"id: {selected_index} has been selected"

            cost_amount += self.cost_f
            cost_amount += self.infos_origin[selected_index]["gt_names"].shape[0] * self.cost_b
            if cost_amount > int(self.current_budget):
                break
            sys.stdout.write(f"cost amount: ({cost_amount}/{self.current_budget})\r")
            sys.stdout.flush()
            selected_index_list.append(selected_index)

        # get the all selected sample ids
        self.selected_index[self.current_budget] = sampled_index_list + selected_index_list
