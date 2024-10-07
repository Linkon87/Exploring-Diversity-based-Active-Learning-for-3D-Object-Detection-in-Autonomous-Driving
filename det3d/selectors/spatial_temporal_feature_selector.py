import os
import sys
import random
import logging
from typing import Dict, List, Optional
from scipy import sparse, spatial
import torch
import numpy as np
from tqdm import tqdm, trange
from det3d import torchie
from det3d.torchie.apis.train import example_to_device
from .base_selector import BaseSelector
from .registry import SELECTORS


@SELECTORS.register_module
class SpatialTemporalFeatureSelector(BaseSelector):
    def __init__(
            self,
            budget: int,
            buffer_file: str,
            dump_file_name: Optional[str] = None,
            infos_origin: List[Dict] = [],
            buffer_path: str = "/home/st2000/data/buffers/feature_pred.pt",
            detector: Optional[torch.nn.Module] = None,
            dataloader: Optional[torch.utils.data.DataLoader] = None,
            logger: Optional[logging.Logger] = None,
            pred: bool = True,
            k: int = 8,
            p: int = 2,
            logs_file: str = "/home/st2000/data/Datasets/nuScenes/train/v1.0-trainval/log.json",
            distance_store_file: str = "/home/st2000/data/buffers/dijkstra_distance_map.npy",
            cost_b: float = 0.04,
            cost_f: float = 0.12,
            lambda_f: float = 1,
            lambda_t: float = 1,
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
        self.logs_file = logs_file
        self.k = k
        self.distance_store_file = distance_store_file
        assert p in [1, 2]
        self.p = p
        self.lambda_f = lambda_f
        self.lambda_t = lambda_t
        self.logger.info(f"lambda_f: {self.lambda_f}")
        self.logger.info(f"lambda_t: {self.lambda_t}")

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
        return distance_map

    def select_samples(self, **kwargs) -> None:
        """feature term"""
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
        feature_distance_map = self.get_feature_distance_map(feats).cpu().numpy()

        """spatial term"""
        if os.path.exists(self.distance_store_file):
            self.logger.info(f"begin to load the distance map from {self.distance_store_file}")
            spatial_distance_map = np.load(self.distance_store_file)
            self.logger.info(f"load the distance map from {self.distance_store_file}")
        else:
            # get the log information of each frame
            logs = torchie.load(self.logs_file)
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
                spatial_distance_map = sparse.csgraph.shortest_path(sparse_distances, directed=False, method="D")

            np.save(self.distance_store_file, spatial_distance_map)
            self.logger.info(f"save the distance map as {self.distance_store_file}")

        """temporal term"""
        num_infos = len(self.infos_origin)
        self.logger.info("begin to calculate the temporal distance map")
        margin = 1e6
        temporal_distance_map = np.ones([num_infos, num_infos]) * margin
        logfile_flag = self.infos_origin[0]["cam_front_path"].split("/")[-1].split("__")[0]
        flag = 0
        max_temporal_distance, count = 0, 0
        logfile2frames = {"0": []}
        for i, info in tqdm(enumerate(self.infos_origin)):
            temp_logfile = info["cam_front_path"].split("/")[-1].split("__")[0]
            if temp_logfile == logfile_flag:
                logfile2frames[str(flag)].append(i)
                count += 1
            else:
                logfile_flag = temp_logfile
                flag += 1
                logfile2frames[str(flag)] = [i]
                if count > max_temporal_distance:
                    max_temporal_distance = count
                count = 1

        for logfile, frames in logfile2frames.items():
            for frame in frames:
                temporal_distance_map[frame, frames] = np.abs(np.array(frames) - frame)
        self.logger.info("get the temporal distance map")

        """combine the spatial and temporal distance"""
        # normalize spatial_distance_map and temporal_distance_map
        spatial_distance_map = 1 - np.exp(-spatial_distance_map)
        temporal_distance_map = 1 - np.exp(-temporal_distance_map)
        feature_distance_map = 1 - np.exp(-feature_distance_map)

        distance_map = np.zeros([num_infos, num_infos])
        for i in trange(len(distance_map)):
            distance_map[i] = spatial_distance_map[i] + self.lambda_t* temporal_distance_map[i] + self.lambda_f * feature_distance_map[
                i]
        # distance_map = spatial_distance_map + temporal_distance_map + self.lambda_f * feature_distance_map

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
        self.logger.info("have selected the beginning index")

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
