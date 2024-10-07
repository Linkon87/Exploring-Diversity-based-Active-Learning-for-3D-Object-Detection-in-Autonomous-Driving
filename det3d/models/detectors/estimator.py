import os
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max
from ..registry import DETECTORS, ESTIMATORS
from .single_stage import SingleStageDetector

try:
    import open3d as o3d
except:
    pass
from det3d.ops.iou3d_nms import boxes_iou3d_gpu
from det3d.ops.pointnet2 import pointnet2_utils
from det3d.core.bbox import geometry
from det3d.core.sampler import preprocess as prep
from det3d.core.bbox import box_torch_ops
from det3d.core.bbox.box_np_ops import center_to_corner_box3d

@ESTIMATORS.register_module
class Estimator(nn.Module):
    def __init__(
        self,
        tasks,
        dim_feat,
    ):
        super().__init__()
        # NOTE: define the estimator
        
        self.dim_feat = dim_feat
        self.num_tasks = len(tasks)
        self.num_classes = sum([len(t["class_names"]) for t in tasks])
        # self.iou_estimator = nn.Sequential(
        #     nn.Linear(9 + dim_feat + self.num_classes, 32),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(True),
        #     nn.Linear(32, 64),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(True),
        # )
        self.iou_estimator = nn.Sequential(
            nn.Linear(9 + self.num_classes, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
        )
        self.iou_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, 1),
        )

    def forward(self, example, return_loss=True, **kwargs):
        # import ipdb; ipdb.set_trace()
        input_features, instance_count, rbboxs, scores, labels = self.extract_points_feature_gpu(example)
    
        pred_ious = self.estimate(input_features, instance_count)
        # loss_rets = self.loss(rbboxs, example, pred_ious)

        bias = 0
        for pred in example["preds"]:
            num_boxes = pred["box3d_lidar"].shape[0]
            pred["scores"] = pred_ious[bias: bias + num_boxes]
            bias += num_boxes

        preds = {
            "rbbox": rbboxs,
            "scores": scores,
            "num_points": instance_count,
            "labels": labels,
            "ious": pred_ious,
        }

        if return_loss:
            loss_rets = self.loss(rbboxs, example, pred_ious)
            return loss_rets
        else:
            # return preds
            return example["preds"]

    def estimate(self, input_features: torch.Tensor, instance_count: torch.Tensor) -> torch.Tensor:
        """estimate the iou for each predicted rotated bounding box

        Args:
            input_features (torch.Tensor, [n, 19]): the encoded features of each inside points for bboxes
            instance_count (torch.Tensor, [n]): the box ids for each points (for max pooling)

        Returns:
            torch.Tensor: predicted iou for each box
        """
        output_iou_feature = self.iou_estimator(input_features) # [n', F]
        output_iou_feature = scatter_max(output_iou_feature, instance_count, 0)[0] # [num_pred, F]-max pooling
        output_iou = torch.sigmoid(self.iou_head(output_iou_feature)).view(-1) # [num_pred]
        return output_iou

    def loss(self, rbboxs: List[torch.Tensor], example: Dict, pred_ious: torch.Tensor) -> Dict:
        device = rbboxs[0].device
        gt_rbboxs = []
        for anno in example["annos"]:
            gt_boxes = np.concatenate(anno["gt_boxes"])
            gt_rbboxs.append(torch.from_numpy(np.concatenate([gt_boxes[:, :6], gt_boxes[:, (-1,)]], 1)).to(device))
        
        gt_ious = []
        for rbbox, gt_rbbox in zip(rbboxs, gt_rbboxs):
            iou = boxes_iou3d_gpu(rbbox, gt_rbbox) # [num_pred', num_gt']
            iou = iou.max(1)[0] # [num_pred']
            gt_ious.append(iou)
        
        # visualization
        # points = example["points"].cpu().numpy()[:, 1:4]
        # o3d_pc = o3d.geometry.PointCloud()
        # o3d_pc.points = o3d.utility.Vector3dVector(points)

        # SAVE_ROOT = "./temp"
        # o3d.io.write_point_cloud(
        #     os.path.join(SAVE_ROOT, f"temp_lidar.ply"), o3d_pc)

        # # visualization
        # o3d_lines = o3d.geometry.LineSet()
        # for index, gt_box in enumerate(gt_rbboxs[0].cpu().numpy()):
        #     box_center = gt_box[:3]
        #     box_dim = gt_box[3:6]
        #     box_angle = gt_box[-1]
        #     o3d_line = get_box(box_angle, box_center, box_dim, [0.5, 0.5, 0.5])
        #     o3d_lines += o3d_line
        # o3d.io.write_line_set(os.path.join(SAVE_ROOT, f"temp_gt_boxes.ply"), o3d_lines)

        # o3d_lines = o3d.geometry.LineSet()
        # for index, box in enumerate(rbboxs[0].cpu().numpy()):
        #     box_center = box[:3]
        #     box_dim = box[3:6]
        #     box_angle = box[-1]
        #     o3d_line = get_box(box_angle, box_center, box_dim, [0.5, 0.5, 0.5])
        #     o3d_lines += o3d_line
        # o3d.io.write_line_set(os.path.join(SAVE_ROOT, f"temp_boxes.ply"), o3d_lines)
        # import ipdb; ipdb.set_trace()
        
        gt_ious = torch.cat(gt_ious) # [num_pred]
        loss = F.binary_cross_entropy(pred_ious, gt_ious).mean()
        loss = [loss] * self.num_tasks # build the multi-task loss and log for cbgs loss API
        rets = {"loss": loss}
        return rets

    def extract_points_feature_gpu(self, example: Dict):
        """
        extract points for each bounding box and assign the scores and labels

        Args:
            example (Dict): input datas
            preds (List[Dict]): prediction
        """
        preds = example["preds"]
        all_points = example["points"]

        # get the features of points by FPN-interpolate
        middle = example["middle"]

        # # import ipdb; ipdb.set_trace()
        # vx_feat, vx_nxyz = tensor2points(middle[0], (-51.2, -51.2, -5.0), voxel_size=(0.2, 0.2, 0.4))
        # p0 = nearest_neighbor_interpolate(all_points, vx_nxyz, vx_feat) # [N, C]

        # vx_feat, vx_nxyz = tensor2points(middle[1], (-51.2, -51.2, -5.0), voxel_size=(0.4, 0.4, 0.8))
        # p1 = nearest_neighbor_interpolate(all_points, vx_nxyz, vx_feat) # [N, C]

        # vx_feat, vx_nxyz = tensor2points(middle[2], (-51.2, -51.2, -5.0), voxel_size=(0.8, 0.8, 1.6))
        # p2 = nearest_neighbor_interpolate(all_points, vx_nxyz, vx_feat) # [N, C]

        # point_feats = torch.cat([p0, p1, p2], dim=-1) # [N, C']
        # dim_feat = point_feats.shape[-1]

        inst_id = 0
        rbboxs = [[] for _ in range(len(preds))] # store rotated bboxes to calculate iou
        scores = [[] for _ in range(len(preds))] # store scores
        labels = [[] for _ in range(len(preds))] # store semantic labels
        input_features = []
        semantic_one_hots = []
        instance_count = []
        preds_processed = [] # save the preds with iou predicted
        for b_i, batch_pred in enumerate(preds):
            batch_pred_processed = {
                "box3d_lidar": [],
                "scores": [],
                "label_preds": [],
                "metadata": batch_pred["metadata"],
            }
            points = all_points[all_points[:, 0] == b_i][:, 1:4] # [N, 3]
            # batch_point_feats = point_feats[all_points[:, 0] == b_i, :] # [N, 3C]
            rbbox_gpu = batch_pred["box3d_lidar"][:, :7] # [M, 7]
            # NOTE expand
            rot_mat_Ts = batch_z_rotation_matrix_torch(rbbox_gpu[:, -1]) # [M, 3, 3]
            expand_factor = 2
            bev_rbbox_gpu = box_torch_ops.center_to_corner_box2d(
                rbbox_gpu[:, :2], rbbox_gpu[:, 3:5] * expand_factor, rbbox_gpu[:, -1]) # [M, 4, 2]
            # NOTE: split to avoid oom
            interval = 20
            times = len(bev_rbbox_gpu) // interval + 1
            pos_mask = []
            for i in range(times):
                bev_rbbox_gpu_interval = bev_rbbox_gpu[i * interval: (i + 1) * interval, ...] # [interval, 4, 2]
                p_mask = geometry.points_in_convex_polygon_torch(points[:, :2], bev_rbbox_gpu_interval) # [N, interval]
                pos_mask.append(p_mask)
            pos_mask = torch.cat(pos_mask, dim=1) # [N, M]
            # pos_mask = geometry.points_in_convex_polygon_torch(points[:, :2], bev_rbbox_gpu) # [N, M]
            top_scores = batch_pred["scores"] # [M]
            semantic_labels = batch_pred["label_preds"] # [M]

            for i, (box_gpu, top_score, semantic_label) in \
                enumerate(zip(rbbox_gpu, top_scores, semantic_labels)):
                # get the points cover by the bev bounding boxes
                instance_mask = pos_mask[:, i] # [N]
                instance_points = points[instance_mask].reshape(-1, 3)
                # instance_features = batch_point_feats[instance_mask].reshape(-1, dim_feat)

                # filter instance points by z-dim of box
                inside_ids = torch.abs(instance_points[:, 2] - box_gpu[2]) <= (box_gpu[5] / 2)
                if inside_ids.sum() == 0: # skip the box that not include point
                    continue
                num_points = inside_ids.sum()
                instance_points = instance_points[inside_ids] # [n, 3]
                # instance_features = instance_features[inside_ids]
                rbboxs[b_i].append(box_gpu)
                scores[b_i].append(top_score)
                labels[b_i].append(semantic_label)
                batch_pred_processed["box3d_lidar"].append(batch_pred["box3d_lidar"][i])
                batch_pred_processed["scores"].append(batch_pred["scores"][i])
                batch_pred_processed["label_preds"].append(batch_pred["label_preds"][i])

                # translate and rotate inside points
                instance_points = instance_points - box_gpu[:3] # [n, 3]
                rot_mat_T = rot_mat_Ts[i] # [3, 3]
                instance_points = instance_points @ rot_mat_T # [n, 3]
                centerness = torch.stack([
                    box_gpu[3]/2 + instance_points[:, 0],
                    box_gpu[3]/2 - instance_points[:, 0],
                    box_gpu[4]/2 + instance_points[:, 1],
                    box_gpu[4]/2 - instance_points[:, 1],
                    box_gpu[5]/2 + instance_points[:, 2],
                    box_gpu[5]/2 - instance_points[:, 2],
                ], 1) # [n, 6]
                instance_count.extend([inst_id] * num_points)
                inst_id += 1
                input_feature = torch.cat([instance_points, centerness], dim=1) # [n, 9 + C']
                # input_feature = torch.cat([instance_points, centerness, instance_features], dim=1) # [n, 9 + C']
                input_features.append(input_feature)
                points_semantic = semantic_label.new_ones(num_points) * semantic_label
                semantic_one_hots.append(F.one_hot(points_semantic, self.num_classes).float()) # [n, 10]

            batch_pred_processed["box3d_lidar"] = torch.stack(batch_pred_processed["box3d_lidar"])
            batch_pred_processed["scores"] = torch.stack(batch_pred_processed["scores"])
            batch_pred_processed["label_preds"] = torch.stack(batch_pred_processed["label_preds"])
            preds_processed.append(batch_pred_processed)
            rbboxs[b_i] = torch.stack(rbboxs[b_i]) # [n', 7]
            scores[b_i] = torch.stack(scores[b_i]) # [n', 7]
            labels[b_i] = torch.stack(labels[b_i]) # [n', 7]
        
        # write the processed prediction
        example["preds"] = preds_processed
        input_features = torch.cat(input_features).float() # [n', 9]
        semantics = torch.cat(semantic_one_hots) # [n', 10]
        input_features = torch.cat([input_features, semantics], dim=1) # [n', 19 + C']
        instance_count = torch.tensor(instance_count, device=input_features.device).long() # [n']

        return input_features, instance_count, rbboxs, scores, labels


    def extract_points_feature(self, example: Dict, preds: List[Dict]):
        """
        extract points for each bounding box and assign the scores and labels

        Args:
            example (Dict): input datas
            preds (List[Dict]): prediction
        """
        all_points = example["points"].cpu().numpy()
        inst_id = 0
        rbboxs = [[] for _ in range(len(preds))] # store rotated bboxes to calculate iou
        scores = [[] for _ in range(len(preds))] # store scores
        labels = [[] for _ in range(len(preds))] # store semantic labels
        input_features = []
        semantic_one_hots = []
        instance_count = []
        for b_i, batch_pred in enumerate(preds):
            points = all_points[all_points[:, 0] == b_i][:, 1:4] # [N, 3]
            rbbox_gpu = batch_pred["box3d_lidar"][:, :7] # [M, 7]
            rbbox = rbbox_gpu.cpu().numpy() # [M, 7]
            top_scores = batch_pred["scores"] # [M]
            semantic_labels = batch_pred["label_preds"] # [M]
            rbbox_corners = center_to_corner_box3d(
                rbbox[:, :3], rbbox[:, 3:6], rbbox[:, -1]) # [M, 8, 3]
            
            mask = prep.mask_points_in_corners(points, rbbox_corners) # [N, M]
            # loop instance 
            for i, (box, box_gpu, top_score, semantic_label) in \
                enumerate(zip(rbbox, rbbox_gpu, top_scores, semantic_labels)):
                instance_mask = mask[:, i] # [N]
                if instance_mask.sum() == 0: # skip the box that not include point
                    continue
                rbboxs[b_i].append(box_gpu)
                scores[b_i].append(top_score)
                labels[b_i].append(semantic_label)
                num_points = instance_mask.sum()
                instance_points = points[instance_mask].reshape(-1, 3) # [n, 3]
                # translate and rotate inside points
                instance_points = instance_points - box[:3] # [n, 3]
                rot_mat_T = z_rotation_matrix(box[-1]) # [3, 3]
                instance_points = instance_points @ rot_mat_T # [n, 3]
                centerness = np.stack([
                    box[3]/2 + instance_points[:, 0],
                    box[3]/2 - instance_points[:, 0],
                    box[4]/2 + instance_points[:, 1],
                    box[4]/2 - instance_points[:, 1],
                    box[5]/2 + instance_points[:, 2],
                    box[5]/2 - instance_points[:, 2],
                ], 1) # [n, 6]
                instance_count.extend([inst_id] * num_points)
                inst_id += 1
                input_feature = np.concatenate([instance_points, centerness], axis=1) # [n, 9]
                input_features.append(input_feature)
                points_semantic = semantic_label.new_ones(num_points) * semantic_label
                semantic_one_hots.append(F.one_hot(points_semantic, self.num_classes).float()) # [n, 10]
            rbboxs[b_i] = torch.stack(rbboxs[b_i]) # [n', 7]
        
        input_features = torch.from_numpy(np.concatenate(input_features)).float().to(semantic_label.device) # [n', 9]
        semantics = torch.cat(semantic_one_hots) # [n', 10]
        input_features = torch.cat([input_features, semantics], dim=1) # [n', 19]
        instance_count = torch.Tensor(instance_count).long().to(input_features.device) # [n']

        return input_features, instance_count, rbboxs, scores, labels


@DETECTORS.register_module
class Estimator(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(Estimator, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )
        # NOTE: define the estimator
        
        self.num_tasks = len(bbox_head.tasks)
        self.num_classes = sum([len(t["class_names"]) for t in bbox_head.tasks])
        self.iou_estimator = nn.Sequential(
            nn.Linear(19, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
        )
        self.iou_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 1),
        )

    def extract_feat(self, data):
        input_features = self.reader(data["features"], data["num_voxels"])
        x = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        with torch.no_grad():
            x = self.extract_feat(data)
            preds = self.bbox_head(x)

        preds = self.bbox_head.predict(example, preds, self.test_cfg)
        input_features, instance_count, rbboxs, scores, labels = self.extract_points_feature(example, preds)
        pred_ious = self.estimate(input_features, instance_count)

        preds = {
            "rbbox": rbbox,
            "scores": scores,
            "num_points": instance_count,
            "labels": labels,
            "ious": pred_ious,
        }

        if return_loss:
            loss_rets = self.loss(rbboxs, example, pred_ious)
            return loss_rets
            # return self.bbox_head.loss(example, preds)
        else:
            return preds
        # else:
        #     return self.bbox_head.predict(example, preds, self.test_cfg)

    def estimate(self, input_features: torch.Tensor, instance_count: torch.Tensor) -> torch.Tensor:
        """estimate the iou for each predicted rotated bounding box

        Args:
            input_features (torch.Tensor, [n, 19]): the encoded features of each inside points for bboxes
            instance_count (torch.Tensor, [n]): the box ids for each points (for max pooling)

        Returns:
            torch.Tensor: predicted iou for each box
        """
        output_iou_feature = self.iou_estimator(input_features) # [n', F]
        output_iou_feature = scatter_max(output_iou_feature, instance_count, 0)[0] # [num_pred, F]-max pooling
        output_iou = torch.sigmoid(self.iou_head(output_iou_feature)).view(-1) # [num_pred]
        return output_iou

    def loss(self, rbboxs: List[torch.Tensor], example: Dict, pred_ious: torch.Tensor) -> Dict:
        device = rbboxs[0].device
        gt_rbboxs = []
        for anno in example["annos"]:
            gt_boxes = np.concatenate(anno["gt_boxes"])
            gt_rbboxs.append(torch.from_numpy(np.concatenate([gt_boxes[:, :6], gt_boxes[:, (-1,)]], 1)).to(device))
        
        gt_ious = []
        for rbbox, gt_rbbox in zip(rbboxs, gt_rbboxs):
            iou = boxes_iou3d_gpu(rbbox, gt_rbbox) # [num_pred', num_gt']
            iou = iou.max(1)[0] # [num_pred']
            gt_ious.append(iou)
        
        gt_ious = torch.cat(gt_ious) # [num_pred]
        loss = F.binary_cross_entropy(pred_ious, gt_ious).mean()
        loss = [loss] * self.num_tasks # build the multi-task loss and log for cbgs loss API
        rets = {"loss": loss}
        return rets


    def extract_points_feature(self, example: Dict, preds: List[Dict]):
        """
        extract points for each bounding box and assign the scores and labels

        Args:
            example (Dict): input datas
            preds (List[Dict]): prediction
        """
        all_points = example["points"].cpu().numpy()
        inst_id = 0
        rbboxs = [[] for _ in range(len(preds))] # store rotated bboxes to calculate iou
        scores = [[] for _ in range(len(preds))] # store scores
        labels = [[] for _ in range(len(preds))] # store semantic labels
        input_features = []
        semantic_one_hots = []
        instance_count = []
        for b_i, batch_pred in enumerate(preds):
            points = all_points[all_points[:, 0] == b_i][:, 1:4] # [N, 3]
            rbbox_gpu = batch_pred["box3d_lidar"][:, :7] # [M, 7]
            rbbox = rbbox_gpu.cpu().numpy() # [M, 7]
            top_scores = batch_pred["scores"] # [M]
            semantic_labels = batch_pred["label_preds"] # [M]
            rbbox_corners = center_to_corner_box3d(
                rbbox[:, :3], rbbox[:, 3:6], rbbox[:, -1]) # [M, 8, 3]
            
            mask = prep.mask_points_in_corners(points, rbbox_corners) # [N, M]
            # loop instance
            for i, (box, box_gpu, top_score, semantic_label) in \
                enumerate(zip(rbbox, rbbox_gpu, top_scores, semantic_labels)):
                instance_mask = mask[:, i] # [N]
                if instance_mask.sum() == 0: # skip the box that not include point
                    continue
                rbboxs[b_i].append(box_gpu)
                scores[b_i].append(top_score)
                labels[b_i].append(semantic_label)
                num_points = instance_mask.sum()
                instance_points = points[instance_mask].reshape(-1, 3) # [n, 3]
                # translate and rotate inside points
                instance_points = instance_points - box[:3] # [n, 3]
                rot_mat_T = z_rotation_matrix(box[-1]) # [3, 3]
                instance_points = instance_points @ rot_mat_T # [n, 3]
                centerness = np.stack([
                    box[3]/2 + instance_points[:, 0],
                    box[3]/2 - instance_points[:, 0],
                    box[4]/2 + instance_points[:, 1],
                    box[4]/2 - instance_points[:, 1],
                    box[5]/2 + instance_points[:, 2],
                    box[5]/2 - instance_points[:, 2],
                ], 1) # [n, 6]
                instance_count.extend([inst_id] * num_points)
                inst_id += 1
                input_feature = np.concatenate([instance_points, centerness], axis=1) # [n, 9]
                input_features.append(input_feature)
                points_semantic = semantic_label.new_ones(num_points) * semantic_label
                semantic_one_hots.append(F.one_hot(points_semantic, self.num_classes).float()) # [n, 10]
            rbboxs[b_i] = torch.stack(rbboxs[b_i]) # [n', 7]
        
        input_features = torch.from_numpy(np.concatenate(input_features)).float().to(semantic_label.device) # [n', 9]
        semantics = torch.cat(semantic_one_hots) # [n', 10]
        input_features = torch.cat([input_features, semantics], dim=1) # [n', 19]
        instance_count = torch.Tensor(instance_count).long().to(input_features.device) # [n']

        return input_features, instance_count, rbboxs, scores, labels


def z_rotation_matrix(angle):
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_T = np.eye(3)

    rot_mat_T[0, 0] = rot_cos
    rot_mat_T[0, 1] = -rot_sin
    rot_mat_T[1, 0] = rot_sin
    rot_mat_T[1, 1] = rot_cos
    return rot_mat_T


def batch_z_rotation_matrix_torch(angle):
    rot_mat_Ts = angle.new_zeros([angle.shape[0], 3, 3]) # [M, 3, 3]
    rot_sin = torch.sin(angle) # [M]
    rot_cos = torch.cos(angle) # [M]

    rot_mat_Ts[:, 0, 0] = rot_cos
    rot_mat_Ts[:, 0, 1] = -rot_sin
    rot_mat_Ts[:, 1, 0] = rot_sin
    rot_mat_Ts[:, 1, 1] = rot_cos
    rot_mat_Ts[:, 2, 2] = 1
    return rot_mat_Ts


def z_rotation_matrix_torch(angle):
    rot_sin = torch.sin(angle)
    rot_cos = torch.cos(angle)
    rot_mat_T = torch.eye(3, device=angle.device)

    rot_mat_T[0, 0] = rot_cos
    rot_mat_T[0, 1] = -rot_sin
    rot_mat_T[1, 0] = rot_sin
    rot_mat_T[1, 1] = rot_cos
    return rot_mat_T

# from SA-SSD: https://github.com/skyhehe123/SA-SSD/blob/master/mmdet/core/bbox/transforms.py
def tensor2points(tensor, offset=(-51.2, -51.2, -5.0), voxel_size=(0.1, 0.1, 0.2)):
    indices = tensor.indices.float()
    offset = torch.Tensor(offset).to(indices.device)
    voxel_size = torch.Tensor(voxel_size).to(indices.device)
    indices[:, 1:] = indices[:, [3, 2, 1]] * voxel_size + offset + 0.5 * voxel_size
    return tensor.features, indices


# from SA-SSD: https://github.com/skyhehe123/SA-SSD/blob/master/mmdet/models/necks/cmn.py
def nearest_neighbor_interpolate(unknown, known, known_feats):
    """three interpolate for unknown from known

    Args:
        unknown (torch.Tensor, [N, 4]): tensor of the bxyz positions of the unknown features
        known (torch.Tensor, [M, 4]): tensor of the bxyz positions of the known features
        known_feats (torch.Tensor, [M, C]): tensor of features to be propigated

    Returns:
        torch.Tensor [N, C]: tensor of the features of the unknown features
    """
    dist, idx = pointnet2_utils.three_nn(unknown, known)
    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=1, keepdim=True)
    weight = dist_recip / norm
    interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)

    return interpolated_feats


def get_box(box_angle, box_center, box_dim, color=[0.5, 0.5, 0.5]):
    rotate_matrix = z_rotation_matrix(box_angle).T
    o3d_box = o3d.geometry.OrientedBoundingBox(box_center, rotate_matrix, box_dim)
    o3d_line = o3d.geometry.LineSet()
    o3d_line = o3d_line.create_from_oriented_bounding_box(o3d_box)
    # o3d_line = o3d_line.paint_uniform_color(np.array(color)[:, np.newaxis])
    return o3d_line

