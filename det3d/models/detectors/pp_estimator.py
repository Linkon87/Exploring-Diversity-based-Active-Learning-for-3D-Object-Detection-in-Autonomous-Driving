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
class PPEstimator(nn.Module):
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
        self.iou_estimator = nn.Sequential(
            nn.Linear(6 + dim_feat+ self.num_classes, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
        )
        self.iou_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 1),
        )

    def forward(self, example, return_loss=True, **kwargs):
        input_features, instance_count, rbboxs, scores, labels = self.extract_pp_features_gpu(example)
    
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
        
        gt_ious = torch.cat(gt_ious) # [num_pred]
        loss = F.binary_cross_entropy(pred_ious, gt_ious).mean()
        loss = [loss] * self.num_tasks # build the multi-task loss and log for cbgs loss API
        rets = {"loss": loss}
        return rets

    def extract_pp_features_gpu(self, example: Dict):
        """
        extract points for each bounding box and assign the scores and labels

        Args:
            example (Dict): input datas
            preds (List[Dict]): prediction
        """
        preds = example["preds"]

        # get the features of points by FPN-interpolate
        middle = example["middle"]

        pp_features = middle[-1] # [B, C, H, W]
        
        pp_features, pp_coords = map2pillars(pp_features, (-51.2, -51.2), (0.8, 0.8)) # [B * H * W, C], [B * H * W, 3]

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
            b_ids = pp_coords[:, 0] == b_i
            pp_coord = pp_coords[b_ids, 1:] # [H * W, 2]
            pp_feature = pp_features[b_ids, :] # [H * W, C]
            dim_feat = pp_feature.shape[-1]

            rbbox_gpu = batch_pred["box3d_lidar"][:, :7] # [M, 7]
            # NOTE expand
            rot_mat_Ts = batch_z_rotation_matrix_torch(rbbox_gpu[:, -1]) # [M, 3, 3]
            expand_factor = 1
            bev_rbbox_gpu = box_torch_ops.center_to_corner_box2d(
                rbbox_gpu[:, :2], rbbox_gpu[:, 3:5] * expand_factor, rbbox_gpu[:, -1]) # [M, 4, 2]
            pos_mask = geometry.points_in_convex_polygon_torch(pp_coord, bev_rbbox_gpu) # [H * W, M]

            top_scores = batch_pred["scores"] # [M]
            semantic_labels = batch_pred["label_preds"] # [M]

            for i, (box_gpu, top_score, semantic_label) in \
                enumerate(zip(rbbox_gpu, top_scores, semantic_labels)):
                # get the points cover by the bev bounding boxes
                pos_ind = pos_mask[:, i].nonzero()[:, 0]

                # NOTE: fix the bugs of targets assignment in bev, while using hybird coordinates,
                #       the `effective_boxes` may not expand enough to cover a grid center,
                #       so we nearest search a grid center as hotspots for this situation
                center = box_gpu[None, :2] # [1, 2]
                dist_to_grid_center = torch.norm(pp_coord - center, dim=1) # [H * W]
                min_ind = torch.argmin(dist_to_grid_center)
                if min_ind not in pos_ind:
                    pos_ind = torch.cat([pos_ind.reshape(-1, 1), min_ind.reshape(-1, 1)],
                                         dim=0).reshape(-1)

                instance_pillars = pp_coord[pos_ind].reshape(-1, 2)
                instance_features = pp_feature[pos_ind].reshape(-1, dim_feat)
                num_pillars = pos_ind.shape[0]

                rbboxs[b_i].append(box_gpu)
                scores[b_i].append(top_score)
                labels[b_i].append(semantic_label)
                batch_pred_processed["box3d_lidar"].append(batch_pred["box3d_lidar"][i])
                batch_pred_processed["scores"].append(batch_pred["scores"][i])
                batch_pred_processed["label_preds"].append(batch_pred["label_preds"][i])

                # translate and rotate inside points
                instance_pillars = instance_pillars - box_gpu[:2] # [n, 2]
                rot_mat_T = rot_mat_Ts[i][:2, :2] # [2, 2]
                instance_pillars = instance_pillars @ rot_mat_T # [n, 2]
                centerness = torch.stack([
                    box_gpu[3]/2 + instance_pillars[:, 0],
                    box_gpu[3]/2 - instance_pillars[:, 0],
                    box_gpu[4]/2 + instance_pillars[:, 1],
                    box_gpu[4]/2 - instance_pillars[:, 1],
                ], 1) # [n, 4]
                instance_count.extend([inst_id] * num_pillars)
                inst_id += 1
                input_feature = torch.cat([instance_pillars, centerness, instance_features], dim=1) # [n, 6 + C']
                input_features.append(input_feature)
                points_semantic = semantic_label.new_ones(num_pillars) * semantic_label
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

# from SA-SSD: https://github.com/skyhehe123/SA-SSD/blob/master/mmdet/core/bbox/transforms.py
def map2pillars(feat, offset=(-51.2, -51.2), pillar_size=(0.1, 0.1)):
    B, C, H, W = feat.shape
    feat = feat.permute(0, 2, 3, 1) # [B, H, W, C]
    feat = feat.reshape(B * H * W, C) # [B * H * W, C]
    coords = feat.new_zeros([B * H * W, 3]) # [B * H * W, 3]
    for i in range(B):
        coords[i * H * W: (i + 1) * H * W, 0] = i
    offset = torch.Tensor(offset).to(coords.device)
    pillar_size = torch.Tensor(pillar_size).to(coords.device)
    coords[:, 1:] = coords[:, 1:] * pillar_size + offset + 0.5 * pillar_size

    return feat, coords


