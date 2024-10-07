import logging
import os
import random
import sys
from typing import Dict, List, Optional
import torch
import numpy as np
import scipy.sparse as sparse
import scipy.spatial as spatial
from tqdm import tqdm
from det3d import torchie
from .base_selector import BaseSelector
from .registry import SELECTORS
from det3d.torchie.fileio import load


@SELECTORS.register_module
class SpatialSelector(BaseSelector):
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
            k: int = 8,
            logs_file: str = "/home/st2000/data/Datasets/nuScenes/train/v1.0-trainval/log.json",
            distance_store_file: str = "/home/st2000/data/buffers/dijkstra_distance_map.npy",
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
        self.logs_file = logs_file
        self.k = k
        self.distance_store_file = distance_store_file

    def select_samples(self, **kwargs) -> None:
        if os.path.exists(self.distance_store_file):
            self.logger.info(f"begin to load the distance map from {self.distance_store_file}")
            distance_map = np.load(self.distance_store_file)
            self.logger.info(f"load the distance map from {self.distance_store_file}")
        else:
            # get the log information of each frame
            logs = load(self.logs_file)
            # build the logfile-to-location map
            log_to_loc = {}
            for l in logs:
                log_to_loc[l["logfile"]] = l["location"].split("-")[-1]

            # get the spatial infos
            scene_to_frame = {}
            frame_to_scene = []
            locations = []
            self.logger.info("load locations")
            for i, info in enumerate(tqdm(self.infos_origin)):
                logfile = info["cam_front_path"].split("/")[-1].split("__")[0]
                scene_id = log_to_loc[logfile]
                if scene_id not in scene_to_frame:
                    scene_to_frame[scene_id] = {}
                cal = info["car_from_global"]
                location = -(cal[:3, 3].T @ cal[:3, :3])
                scene_to_frame[scene_id][i] = {
                    "location": location,
                }
                frame_to_scene.append(scene_id)
                locations.append(location[:2])
            frame_to_scene = np.array(frame_to_scene)  # [num_infos]
            locations = np.stack(locations)  # [num_infos, 2]
            # construct Euclidean distance map
            num_infos = len(self.infos_origin)
            self.logger.info("begin to calculate the sparse distance map")
            sparse_distances = np.zeros([num_infos, num_infos])
            # knn distance graph
            tree = spatial.cKDTree(locations)
            knn_distances, knn_ids = tree.query(locations, self.k + 1)
            for self_id, (neighbor_distances, neighbor_ids) in tqdm(enumerate(zip(knn_distances, knn_ids))):
                sparse_distances[self_id, neighbor_ids] = neighbor_distances
                sparse_distances[neighbor_ids, self_id] = neighbor_distances

            with torchie.Timer("dijkstra construction time:"):
                distance_map = sparse.csgraph.shortest_path(sparse_distances, directed=False, method="D")

            np.save(self.distance_store_file, distance_map)
            self.logger.info(f"save the distance map as {self.distance_store_file}")

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
