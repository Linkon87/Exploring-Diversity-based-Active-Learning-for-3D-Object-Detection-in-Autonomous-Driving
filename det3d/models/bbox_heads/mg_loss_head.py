import logging
from collections import defaultdict
from enum import Enum

import numpy as np
import torch
from det3d.core import box_torch_ops
from det3d.models.builder import build_loss
from det3d.models.losses import metrics
from det3d.torchie.cnn import constant_init, kaiming_init
from det3d.torchie.trainer import load_checkpoint
from det3d.ops.iou3d_nms import boxes_iou3d_gpu
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from .. import builder
from ..losses import accuracy, WeightedSmoothL1Loss
from ..registry import HEADS


def one_hot_f(tensor, depth, dim=-1, on_value=1.0, dtype=torch.float32):
    tensor_onehot = torch.zeros(
        *list(tensor.shape), depth, dtype=dtype, device=tensor.device
    )
    tensor_onehot.scatter_(dim, tensor.unsqueeze(dim).long(), on_value)
    return tensor_onehot


def add_sin_difference(boxes1, boxes2):
    rad_pred_encoding = torch.sin(boxes1[..., -1:]) * torch.cos(boxes2[..., -1:])
    rad_tg_encoding = torch.cos(boxes1[..., -1:]) * torch.sin(boxes2[..., -1:])
    boxes1 = torch.cat([boxes1[..., :-1], rad_pred_encoding], dim=-1)
    boxes2 = torch.cat([boxes2[..., :-1], rad_tg_encoding], dim=-1)
    return boxes1, boxes2


def _get_pos_neg_loss(cls_loss, labels):
    # cls_loss: [N, num_anchors, num_class]
    # labels: [N, num_anchors]
    batch_size = cls_loss.shape[0]
    if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2:
        cls_pos_loss = (labels > 0).type_as(cls_loss) * cls_loss.view(batch_size, -1)
        cls_neg_loss = (labels == 0).type_as(cls_loss) * cls_loss.view(batch_size, -1)
        cls_pos_loss = cls_pos_loss.sum() / batch_size
        cls_neg_loss = cls_neg_loss.sum() / batch_size
    else:
        cls_pos_loss = cls_loss[..., 1:].sum() / batch_size
        cls_neg_loss = cls_loss[..., 0].sum() / batch_size
    return cls_pos_loss, cls_neg_loss


def limit_period(val, offset=0.5, period=np.pi):
    return val - torch.floor(val / period + offset) * period


def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0.0):
    batch_size = reg_targets.shape[0]
    anchors = anchors.view(batch_size, -1, anchors.shape[-1])
    rot_gt = reg_targets[..., -1] + anchors[..., -1]
    #dir_cls_targets = ((rot_gt - dir_offset) > 0).long()
    dir_cls_targets = (limit_period(rot_gt - dir_offset, 0.5, np.pi * 2) >
                       0).long()
    if one_hot:
        dir_cls_targets = one_hot_f(dir_cls_targets, 2, dtype=anchors.dtype)
    return dir_cls_targets


def create_loss(
    loc_loss_ftor,
    cls_loss_ftor,
    box_preds,
    cls_preds,
    cls_targets,
    cls_weights,
    reg_targets,
    reg_weights,
    num_class,
    encode_background_as_zeros=True,
    encode_rad_error_by_sin=True,
    bev_only=False,
    box_code_size=9,
):
    batch_size = int(box_preds.shape[0])

    if bev_only:
        box_preds = box_preds.view(batch_size, -1, box_code_size - 2)
    else:
        box_preds = box_preds.view(batch_size, -1, box_code_size)

    if encode_background_as_zeros:
        cls_preds = cls_preds.view(batch_size, -1, num_class)
    else:
        cls_preds = cls_preds.view(batch_size, -1, num_class + 1)

    cls_targets = cls_targets.squeeze(-1)
    one_hot_targets = one_hot_f(cls_targets, depth=num_class + 1, dtype=box_preds.dtype)
    if encode_background_as_zeros:
        one_hot_targets = one_hot_targets[..., 1:]

    if encode_rad_error_by_sin:
        # sin(a - b) = sinacosb-cosasinb
        box_preds_diff, reg_targets_diff = add_sin_difference(box_preds, reg_targets)
        loc_losses = loc_loss_ftor(box_preds_diff, reg_targets_diff, weights=reg_weights)  # [N, M]
    else:
        loc_losses = loc_loss_ftor(box_preds, reg_targets, weights=reg_weights)  # [N, M]

    cls_losses = cls_loss_ftor(
        cls_preds, one_hot_targets, weights=cls_weights
    )  # [N, M]

    return loc_losses, cls_losses


class LossNormType(Enum):
    NormByNumPositives = "norm_by_num_positives"
    NormByNumExamples = "norm_by_num_examples"
    NormByNumPosNeg = "norm_by_num_pos_neg"
    DontNorm = "dont_norm"


@HEADS.register_module
class LossHead(nn.Module):
    def __init__(
        self,
        num_input,
        num_pred,
        num_cls,
        num_loss,
        use_dir=False,
        num_dir=0,
        header=True,
        name="",
        focal_loss_init=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_dir = use_dir

        self.conv_box = nn.Conv2d(num_input, num_pred, 1)
        self.conv_cls = nn.Conv2d(num_input, num_cls, 1)
        self.mlp_loss = nn.Sequential(
            nn.Conv2d(num_input, num_input // 2, 1, 1),
            nn.BatchNorm2d(num_input // 2),
            nn.ReLU(),
            nn.Conv2d(num_input // 2, num_loss, 1, 1)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        if self.use_dir:
            self.conv_dir = nn.Conv2d(num_input, num_dir, 1)

    def forward(self, x, finetune=False):
        ret_list = []
        if finetune:
            with torch.no_grad():
                box_preds = self.conv_box(x).permute(0, 2, 3, 1).contiguous()
                cls_preds = self.conv_cls(x).permute(0, 2, 3, 1).contiguous()
        else:
            box_preds = self.conv_box(x).permute(0, 2, 3, 1).contiguous()
            cls_preds = self.conv_cls(x).permute(0, 2, 3, 1).contiguous()
        B, C, H, W = x.shape
        x_temp = self.gap(x)
        loss_preds = self.mlp_loss(x_temp)
        loss_preds = loss_preds.view(B, -1)
        loss_preds = self.mlp_loss(x_temp)
        ret_dict = {"box_preds": box_preds, "cls_preds": cls_preds, "loss_preds": loss_preds}
        if self.use_dir:
            dir_preds = self.conv_dir(x).permute(0, 2, 3, 1).contiguous()
            ret_dict["dir_cls_preds"] = dir_preds

        return ret_dict


@HEADS.register_module
class MultiGroupLossHead(nn.Module):
    def __init__(
        self,
        mode="3d",
        in_channels=[128, ],
        norm_cfg=None,
        tasks=[],
        weights=[],
        num_classes=[1, ],
        box_coder=None,
        with_cls=True,
        with_reg=True,
        reg_class_agnostic=False,
        encode_background_as_zeros=True,
        loss_norm=dict(
            type="NormByNumPositives", pos_class_weight=1.0, neg_class_weight=1.0,
        ),
        loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0,),
        use_sigmoid_score=True,
        loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0,),
        encode_rad_error_by_sin=True,
        loss_aux=None,
        direction_offset=0.0,
        name="rpn",
        logger=None,
    ):
        super().__init__()

        assert with_cls or with_reg

        num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.num_anchor_per_locs = [2 * n for n in num_classes]

        self.box_coder = box_coder
        box_code_sizes = [box_coder.code_size] * len(num_classes)

        self.with_cls = with_cls
        self.with_reg = with_reg
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.reg_class_agnostic = reg_class_agnostic
        self.encode_rad_error_by_sin = encode_rad_error_by_sin
        self.encode_background_as_zeros = encode_background_as_zeros
        self.use_sigmoid_score = use_sigmoid_score

        # to work with the angle vector encoding
        self.box_n_dim = self.box_coder.code_size
        self.anchor_dim = self.box_coder.n_dim
        # self.box_n_dim = self.box_coder.n_dim

        self.loss_cls = build_loss(loss_cls)
        self.loss_reg = build_loss(loss_bbox)
        if loss_aux is not None:
            self.loss_aux = build_loss(loss_aux)

        self.loss_norm = loss_norm

        if not logger:
            logger = logging.getLogger("MultiGroupHead")
        self.logger = logger

        self.dcn = None
        self.zero_init_residual = False

        self.use_direction_classifier = loss_aux is not None
        if loss_aux:
            self.direction_offset = direction_offset

        self.bev_only = True if mode == "bev" else False

        num_losses = []
        num_clss = []
        num_preds = []
        num_dirs = []

        for num_c, num_a, box_cs in zip(
            num_classes, self.num_anchor_per_locs, box_code_sizes
        ):
            if self.encode_background_as_zeros:
                num_cls = num_a * num_c
            else:
                num_cls = num_a * (num_c + 1)
            num_clss.append(num_cls)

            if self.bev_only:
                num_pred = num_a * (box_cs - 2)
            else:
                num_pred = num_a * box_cs
            num_preds.append(num_pred)

            if self.use_direction_classifier:
                num_dir = num_a * 2
                num_dirs.append(num_dir)
            else:
                num_dir = None

            num_losses.append(1)

        logger.info(
            f"num_classes: {num_classes}, num_preds: {num_preds}, num_dirs: {num_dirs}"
        )

        self.tasks = nn.ModuleList()
        for task_id, (num_pred, num_cls, num_loss) in enumerate(zip(num_preds, num_clss, num_losses)):
            self.tasks.append(
                LossHead(
                    in_channels,
                    num_pred,
                    num_cls,
                    num_loss,
                    use_dir=self.use_direction_classifier,
                    num_dir=num_dirs[task_id]
                    if self.use_direction_classifier
                    else None,
                    header=False,
                )
            )

        logger.info("Finish MultiGroupHead Initialization")

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(m, "conv2_offset"):
                        constant_init(m.conv2_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError("pretrained must be a str or None")

    def forward(self, x, finetune=False):
        ret_dicts = []
        for task in self.tasks:
            ret_dicts.append(task(x, finetune))

        return ret_dicts

    def prepare_loss_weights(
        self,
        labels,
        loss_norm=dict(
            type="NormByNumPositives", pos_cls_weight=1.0, neg_cls_weight=1.0,
        ),
        dtype=torch.float32,
    ):
        loss_norm_type = getattr(LossNormType, loss_norm["type"])
        pos_cls_weight = loss_norm["pos_cls_weight"]
        neg_cls_weight = loss_norm["neg_cls_weight"]

        cared = labels >= 0
        # cared: [N, num_anchors]
        positives = labels > 0
        negatives = labels == 0
        negative_cls_weights = negatives.type(dtype) * neg_cls_weight
        cls_weights = negative_cls_weights + pos_cls_weight * positives.type(dtype)
        reg_weights = positives.type(dtype)
        if loss_norm_type == LossNormType.NormByNumExamples:
            num_examples = cared.type(dtype).sum(1, keepdim=True)
            num_examples = torch.clamp(num_examples, min=1.0)
            cls_weights /= num_examples
            bbox_normalizer = positives.sum(1, keepdim=True).type(dtype)
            reg_weights /= torch.clamp(bbox_normalizer, min=1.0)
        elif loss_norm_type == LossNormType.NormByNumPositives:
            pos_normalizer = positives.sum(1, keepdim=True).type(dtype)
            reg_weights /= torch.clamp(pos_normalizer, min=1.0)
            cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        elif loss_norm_type == LossNormType.NormByNumPosNeg:
            pos_neg = torch.stack([positives, negatives], dim=-1).type(dtype)
            normalizer = pos_neg.sum(1, keepdim=True)  # [N, 1, 2]
            cls_normalizer = (pos_neg * normalizer).sum(-1)  # [N, M]
            cls_normalizer = torch.clamp(cls_normalizer, min=1.0)
            # cls_normalizer will be pos_or_neg_weight/num_pos_or_neg
            normalizer = torch.clamp(normalizer, min=1.0)
            reg_weights /= normalizer[:, 0:1, 0]
            cls_weights /= cls_normalizer
        elif loss_norm_type == LossNormType.DontNorm:  # support ghm loss
            pos_normalizer = positives.sum(1, keepdim=True).type(dtype)
            reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        else:
            raise ValueError(f"unknown loss norm type. available: {list(LossNormType)}")
        return cls_weights, reg_weights, cared

    def compute_loss_loss(self, loss_gt, loss_preds, batch_size):
        sub = torch.abs(loss_gt - loss_preds.sum()) / batch_size
        return sub

    def loss(self, example, preds_dicts, **kwargs):

        voxels = example["voxels"]
        num_points = example["num_points"]
        coors = example["coordinates"]
        batch_anchors = example["anchors"]
        batch_size_device = batch_anchors[0].shape[0]

        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            losses = dict()

            num_class = self.num_classes[task_id]

            box_preds = preds_dict["box_preds"]
            cls_preds = preds_dict["cls_preds"]
            loss_preds = preds_dict["loss_preds"]

            labels = example["labels"][task_id]
            if kwargs.get("mode", False):
                reg_targets = example["reg_targets"][task_id][:, :, [0, 1, 3, 4, 6]]
                reg_targets_left = example["reg_targets"][task_id][:, :, [2, 5]]
            else:
                reg_targets = example["reg_targets"][task_id]

            cls_weights, reg_weights, cared = self.prepare_loss_weights(
                labels, loss_norm=self.loss_norm, dtype=torch.float32,
            )
            cls_targets = labels * cared.type_as(labels)
            cls_targets = cls_targets.unsqueeze(-1)

            loc_loss, cls_loss = create_loss(
                self.loss_reg,
                self.loss_cls,
                box_preds,
                cls_preds,
                cls_targets,
                cls_weights,
                reg_targets,
                reg_weights,
                num_class,
                self.encode_background_as_zeros,
                self.encode_rad_error_by_sin,
                bev_only=self.bev_only,
                box_code_size=self.box_n_dim,
            )

            # iou_loss_reduced = iou_loss.sum() / batch_size_device
            # iou_loss_reduced *= self.loss_iou._loss_weight

            loc_loss_reduced = loc_loss.sum() / batch_size_device
            loc_loss_reduced *= self.loss_reg._loss_weight
            cls_pos_loss, cls_neg_loss = _get_pos_neg_loss(cls_loss, labels)
            cls_pos_loss /= self.loss_norm["pos_cls_weight"]
            cls_neg_loss /= self.loss_norm["neg_cls_weight"]
            cls_loss_reduced = cls_loss.sum() / batch_size_device
            cls_loss_reduced *= self.loss_cls._loss_weight
            loss_loss_reduced = self.compute_loss_loss(
                loc_loss.sum() + cls_loss.sum(), loss_preds, batch_size_device)

            loss = loss_loss_reduced + loc_loss_reduced + cls_loss_reduced

            if self.use_direction_classifier:
                dir_targets = get_direction_target(
                    example["anchors"][task_id],
                    reg_targets,
                    dir_offset=self.direction_offset,
                )
                dir_logits = preds_dict["dir_cls_preds"].view(batch_size_device, -1, 2)
                weights = (labels > 0).type_as(dir_logits)
                weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
                dir_loss = self.loss_aux(dir_logits, dir_targets, weights=weights)
                dir_loss = dir_loss.sum() / batch_size_device
                loss += dir_loss * self.loss_aux._loss_weight

                # losses['loss_aux'] = dir_loss

            loc_loss_elem = [
                loc_loss[:, :, i].sum() / batch_size_device
                for i in range(loc_loss.shape[-1])
            ]
            ret = {
                "loss": loss,
                "cls_pos_loss": cls_pos_loss.detach().cpu(),
                "cls_neg_loss": cls_neg_loss.detach().cpu(),
                "dir_loss_reduced": dir_loss.detach().cpu()
                if self.use_direction_classifier
                else torch.tensor(0),
                "loss_loss_reduced": loss_loss_reduced.detach().cpu().mean(),
                "cls_loss_reduced": cls_loss_reduced.detach().cpu().mean(),
                "loc_loss_reduced": loc_loss_reduced.detach().cpu().mean(),
                "loc_loss_elem": [elem.detach().cpu() for elem in loc_loss_elem],
                "num_pos": (labels > 0)[0].sum(),
                "num_neg": (labels == 0)[0].sum(),
            }

            # self.rpn_acc.clear()
            # losses['acc'] = self.rpn_acc(
            #     example['labels'][task_id],
            #     cls_preds,
            #     cared,
            # )

            # losses['pr'] = {}
            # self.rpn_pr.clear()
            # prec, rec = self.rpn_pr(
            #     example['labels'][task_id],
            #     cls_preds,
            #     cared,
            # )
            # for i, thresh in enumerate(self.rpn_pr.thresholds):
            #     losses["pr"][f"prec@{int(thresh*100)}"] = float(prec[i])
            #     losses["pr"][f"rec@{int(thresh*100)}"] = float(rec[i])

            rets.append(ret)
        """convert batch-key to key-batch
        """
        rets_merged = defaultdict(list)
        for ret in rets:
            for k, v in ret.items():
                rets_merged[k].append(v)

        return rets_merged

    def predict(self, example, preds_dicts, test_cfg, **kwargs):
        """start with v1.6.0, this function don't contain any kitti-specific code.
        Returns:
            predict: list of pred_dict.
            pred_dict: {
                box3d_lidar: [N, 7] 3d box.
                scores: [N]
                label_preds: [N]
                metadata: meta-data which contains dataset-specific information.
                    for kitti, it contains image idx (label idx),
                    for nuscenes, sample_token is saved in it.
            }
        """
        voxels = example["voxels"]
        num_points = example["num_points"]
        coors = example["coordinates"]
        batch_anchors = example["anchors"]
        batch_size_device = batch_anchors[0].shape[0]
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            batch_size = batch_anchors[task_id].shape[0]

            if "metadata" not in example or len(example["metadata"]) == 0:
                meta_list = [None] * batch_size
            else:
                meta_list = example["metadata"]

            batch_task_anchors = example["anchors"][task_id].view(
                batch_size, -1, self.anchor_dim
            )

            if "anchors_mask" not in example:
                batch_anchors_mask = [None] * batch_size
            else:
                batch_anchors_mask = example["anchors_mask"][task_id].view(
                    batch_size, -1
                )

            batch_box_preds = preds_dict["box_preds"]
            batch_cls_preds = preds_dict["cls_preds"]
            batch_iou_preds = preds_dict["iou_preds"]

            if isinstance(self.loss_iou, WeightedSmoothL1Loss):
                batch_iou_preds = batch_iou_preds * \
                    self.iou_norm.get("std") + self.iou_norm.get("mean")
                batch_iou_preds = torch.clamp(batch_iou_preds, min=0, max=1)
            else:
                batch_iou_preds = torch.sigmoid(batch_iou_preds)

            if self.bev_only:
                box_ndim = self.box_n_dim - 2
            else:
                box_ndim = self.box_n_dim

            if kwargs.get("mode", False):
                batch_box_preds_base = batch_box_preds.view(batch_size, -1, box_ndim)
                batch_box_preds = batch_task_anchors.clone()
                batch_box_preds[:, :, [0, 1, 3, 4, 6]] = batch_box_preds_base
            else:
                batch_box_preds = batch_box_preds.view(batch_size, -1, box_ndim)

            num_class_with_bg = self.num_classes[task_id]

            if not self.encode_background_as_zeros:
                num_class_with_bg = self.num_classes[task_id] + 1

            batch_cls_preds = batch_cls_preds.view(batch_size, -1, num_class_with_bg)
            batch_iou_preds = batch_iou_preds.view(batch_size, -1, 1)

            batch_reg_preds = self.box_coder.decode_torch(
                batch_box_preds[:, :, : self.box_coder.code_size], batch_task_anchors
            )

            if self.use_direction_classifier:
                batch_dir_preds = preds_dict["dir_cls_preds"]
                batch_dir_preds = batch_dir_preds.view(batch_size, -1, 2)
            else:
                batch_dir_preds = [None] * batch_size
            rets.append(
                self.get_task_detections(
                    task_id,
                    num_class_with_bg,
                    test_cfg,
                    batch_iou_preds,
                    batch_cls_preds,
                    batch_reg_preds,
                    batch_dir_preds,
                    batch_anchors_mask,
                    meta_list,
                )
            )
        # Merge branches results
        num_tasks = len(rets)
        ret_list = []
        # len(rets) == task num
        # len(rets[0]) == batch_size
        num_preds = len(rets)
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            ret = {}
            for k in rets[0][i].keys():
                if k in ["box3d_lidar", "scores"]:
                    ret[k] = torch.cat([ret[i][k] for ret in rets])
                elif k in ["label_preds"]:
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    ret[k] = torch.cat([ret[i][k] for ret in rets])
                elif k == "metadata":
                    # metadata
                    ret[k] = rets[0][i][k]
            ret_list.append(ret)

        return ret_list

    def get_task_detections(
        self,
        task_id,
        num_class_with_bg,
        test_cfg,
        batch_iou_preds,
        batch_cls_preds,
        batch_reg_preds,
        batch_dir_preds=None,
        batch_anchors_mask=None,
        meta_list=None,
    ):
        predictions_dicts = []
        post_center_range = test_cfg.post_center_limit_range
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=batch_reg_preds.dtype,
                device=batch_reg_preds.device,
            )

        for box_preds, cls_preds, iou_preds, dir_preds, a_mask, meta in zip(
            batch_reg_preds,
            batch_cls_preds,
            batch_iou_preds,
            batch_dir_preds,
            batch_anchors_mask,
            meta_list,
        ):
            if a_mask is not None:
                box_preds = box_preds[a_mask]
                cls_preds = cls_preds[a_mask]
                iou_preds = iou_preds[a_mask]

            box_preds = box_preds.float()
            cls_preds = cls_preds.float()
            iou_preds = iou_preds.float()

            if self.use_direction_classifier:
                if a_mask is not None:
                    dir_preds = dir_preds[a_mask]
                dir_labels = torch.max(dir_preds, dim=-1)[1]

            if self.encode_background_as_zeros:
                # this don't support softmax
                assert self.use_sigmoid_score is True
                total_scores = torch.sigmoid(cls_preds)
                total_scores = total_scores  # * iou_preds
            else:
                # encode background as first element in one-hot vector
                if self.use_sigmoid_score:
                    total_scores = torch.sigmoid(cls_preds)[..., 1:]
                    total_scores = total_scores  # * iou_preds
                else:
                    total_scores = F.softmax(cls_preds, dim=-1)[..., 1:]
                    total_scores = total_scores  # * iou_preds

            # Apply NMS in birdeye view
            if test_cfg.nms.use_rotate_nms:
                nms_func = box_torch_ops.rotate_nms
            else:
                nms_func = box_torch_ops.nms

            feature_map_size_prod = (
                batch_reg_preds.shape[1] // self.num_anchor_per_locs[task_id]
            )

            if test_cfg.nms.use_multi_class_nms:
                assert self.encode_background_as_zeros is True
                boxes_for_nms = box_preds[:, [0, 1, 3, 4, -1]]
                if not test_cfg.nms.use_rotate_nms:
                    box_preds_corners = box_torch_ops.center_to_corner_box2d(
                        boxes_for_nms[:, :2], boxes_for_nms[:, 2:4], boxes_for_nms[:, 4]
                    )
                    boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                        box_preds_corners
                    )

                selected_boxes, selected_labels, selected_scores = [], [], []
                selected_dir_labels = []

                scores = total_scores
                boxes = boxes_for_nms
                selected_per_class = []
                score_threshs = [test_cfg.score_threshold] * self.num_classes[task_id]
                pre_max_sizes = [test_cfg.nms.nms_pre_max_size] * self.num_classes[
                    task_id
                ]
                post_max_sizes = [test_cfg.nms.nms_post_max_size] * self.num_classes[
                    task_id
                ]
                iou_thresholds = [test_cfg.nms.nms_iou_threshold] * self.num_classes[
                    task_id
                ]

                for class_idx, score_thresh, pre_ms, post_ms, iou_th in zip(
                    range(self.num_classes[task_id]),
                    score_threshs,
                    pre_max_sizes,
                    post_max_sizes,
                    iou_thresholds,
                ):
                    self._nms_class_agnostic = False
                    if self._nms_class_agnostic:
                        class_scores = total_scores.view(
                            feature_map_size_prod, -1, self.num_classes[task_id]
                        )[..., class_idx]
                        class_scores = class_scores.contiguous().view(-1)
                        class_boxes_nms = boxes.view(-1, boxes_for_nms.shape[-1])
                        class_boxes = box_preds
                        class_dir_labels = dir_labels
                    else:
                        # anchors_range = self.target_assigner.anchors_range(class_idx)
                        anchors_range = self.target_assigners[task_id].anchors_range
                        class_scores = total_scores.view(
                            -1, self._num_classes[task_id]
                        )[anchors_range[0]: anchors_range[1], class_idx]
                        class_boxes_nms = boxes.view(-1, boxes_for_nms.shape[-1])[
                            anchors_range[0]: anchors_range[1], :
                        ]
                        class_scores = class_scores.contiguous().view(-1)
                        class_boxes_nms = class_boxes_nms.contiguous().view(
                            -1, boxes_for_nms.shape[-1]
                        )
                        class_boxes = box_preds.view(-1, box_preds.shape[-1])[
                            anchors_range[0]: anchors_range[1], :
                        ]
                        class_boxes = class_boxes.contiguous().view(
                            -1, box_preds.shape[-1]
                        )
                        if self.use_direction_classifier:
                            class_dir_labels = dir_labels.view(-1)[
                                anchors_range[0]: anchors_range[1]
                            ]
                            class_dir_labels = class_dir_labels.contiguous().view(-1)
                    if score_thresh > 0.0:
                        class_scores_keep = class_scores >= score_thresh
                        if class_scores_keep.shape[0] == 0:
                            selected_per_class.append(None)
                            continue
                        class_scores = class_scores[class_scores_keep]
                    if class_scores.shape[0] != 0:
                        if score_thresh > 0.0:
                            class_boxes_nms = class_boxes_nms[class_scores_keep]
                            class_boxes = class_boxes[class_scores_keep]
                            class_dir_labels = class_dir_labels[class_scores_keep]
                        keep = nms_func(
                            class_boxes_nms, class_scores, pre_ms, post_ms, iou_th
                        )
                        if keep.shape[0] != 0:
                            selected_per_class.append(keep)
                        else:
                            selected_per_class.append(None)
                    else:
                        selected_per_class.append(None)
                    selected = selected_per_class[-1]

                    if selected is not None:
                        selected_boxes.append(class_boxes[selected])
                        selected_labels.append(
                            torch.full(
                                [class_boxes[selected].shape[0]],
                                class_idx,
                                dtype=torch.int64,
                                device=box_preds.device,
                            )
                        )
                        if self.use_direction_classifier:
                            selected_dir_labels.append(class_dir_labels[selected])
                        selected_scores.append(class_scores[selected])
                    # else:
                    #     selected_boxes.append(torch.Tensor([], device=class_boxes.device))
                    #     selected_labels.append(torch.Tensor([], device=box_preds.device))
                    #     selected_scores.append(torch.Tensor([], device=class_scores.device))
                    #     if self.use_direction_classifier:
                    #         selected_dir_labels.append(torch.Tensor([], device=class_dir_labels.device))

                selected_boxes = torch.cat(selected_boxes, dim=0)
                selected_labels = torch.cat(selected_labels, dim=0)
                selected_scores = torch.cat(selected_scores, dim=0)
                if self.use_direction_classifier:
                    selected_dir_labels = torch.cat(selected_dir_labels, dim=0)

            else:
                # get highest score per prediction, than apply nms
                # to remove overlapped box.
                if num_class_with_bg == 1:
                    top_scores = total_scores.squeeze(-1)
                    top_labels = torch.zeros(
                        total_scores.shape[0],
                        device=total_scores.device,
                        dtype=torch.long,
                    )

                else:
                    top_scores, top_labels = torch.max(total_scores, dim=-1)

                if test_cfg.score_threshold > 0.0:
                    thresh = torch.tensor(
                        [test_cfg.score_threshold], device=total_scores.device
                    ).type_as(total_scores)
                    top_scores_keep = top_scores >= thresh
                    top_scores = top_scores.masked_select(top_scores_keep)

                if top_scores.shape[0] != 0:
                    if test_cfg.score_threshold > 0.0:
                        box_preds = box_preds[top_scores_keep]
                        if self.use_direction_classifier:
                            dir_labels = dir_labels[top_scores_keep]
                        top_labels = top_labels[top_scores_keep]
                    boxes_for_nms = box_preds[:, [0, 1, 3, 4, -1]]
                    if not test_cfg.nms.use_rotate_nms:
                        box_preds_corners = box_torch_ops.center_to_corner_box2d(
                            boxes_for_nms[:, :2],
                            boxes_for_nms[:, 2:4],
                            boxes_for_nms[:, 4],
                        )
                        boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                            box_preds_corners
                        )
                    # the nms in 3d detection just remove overlap boxes.
                    selected = nms_func(
                        boxes_for_nms,
                        top_scores,
                        pre_max_size=test_cfg.nms.nms_pre_max_size,
                        post_max_size=test_cfg.nms.nms_post_max_size,
                        iou_threshold=test_cfg.nms.nms_iou_threshold,
                    )
                else:
                    selected = []
                # if selected is not None:
                selected_boxes = box_preds[selected]
                if self.use_direction_classifier:
                    selected_dir_labels = dir_labels[selected]
                selected_labels = top_labels[selected]
                selected_scores = top_scores[selected]

            # finally generate predictions.
            # self.logger.info(f"selected boxes: {selected_boxes.shape}")
            if selected_boxes.shape[0] != 0:
                # self.logger.info(f"result not none~ Selected boxes: {selected_boxes.shape}")
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                if self.use_direction_classifier:
                    dir_labels = selected_dir_labels
                    opp_labels = (
                        (box_preds[..., -1] - self.direction_offset) > 0
                    ) ^ dir_labels.bool()
                    box_preds[..., -1] += torch.where(
                        opp_labels,
                        torch.tensor(np.pi).type_as(box_preds),
                        torch.tensor(0.0).type_as(box_preds),
                    )
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                if post_center_range is not None:
                    mask = (final_box_preds[:, :3] >= post_center_range[:3]).all(1)
                    mask &= (final_box_preds[:, :3] <= post_center_range[3:]).all(1)
                    predictions_dict = {
                        "box3d_lidar": final_box_preds[mask],
                        "scores": final_scores[mask],
                        "label_preds": label_preds[mask],
                        "metadata": meta,
                    }
                else:
                    predictions_dict = {
                        "box3d_lidar": final_box_preds,
                        "scores": final_scores,
                        "label_preds": label_preds,
                        "metadata": meta,
                    }
            else:
                dtype = batch_reg_preds.dtype
                device = batch_reg_preds.device
                predictions_dict = {
                    "box3d_lidar": torch.zeros([0, self.anchor_dim], dtype=dtype, device=device),
                    "scores": torch.zeros([0], dtype=dtype, device=device),
                    "label_preds": torch.zeros(
                        [0], dtype=top_labels.dtype, device=device
                    ),
                    "metadata": meta,
                }

            predictions_dicts.append(predictions_dict)

        return predictions_dicts
