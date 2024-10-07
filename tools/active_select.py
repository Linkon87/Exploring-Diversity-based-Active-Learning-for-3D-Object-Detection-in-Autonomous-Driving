import os
import time
import random
import logging
import argparse
from typing import Dict
import numpy as np
import torch
from det3d import __version__, torchie
from det3d.torchie import Config
from det3d.torchie.apis import (
    get_root_logger,
)
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.selectors import build_selector, BaseSelector
from det3d.torchie.trainer import load_checkpoint
from torch.nn.parallel import DistributedDataParallel
from det3d.torchie.fileio import load


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("--config", help="train config file path",default='examples/active/cbgs_uwe.py')
    parser.add_argument("--work_dir", help="the dir to save logs and models")
    parser.add_argument(
        "--checkpoint", default='work_dir/NUSC_CBGS_random_budget_600_20221107-003902/latest.pth',help="the dir to checkpoint which the model read from"
    )
    parser.add_argument(
        "--txt_result",
        type=bool,
        default=False,
        help="whether to save results to standard KITTI format of txt type",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=2,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=2400,
        help="number of samples for active learning",
    )
    parser.add_argument(
        "--pred",
        action="store_true",
        help="get the predictions and save as buffer",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        # default="none",
        default="pytorch",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def init_sample_dataset(cfg: Dict, logger: logging.Logger) -> None:
    buffer = {"0": []}
    torchie.dump(buffer, cfg.selector.buffer_file, indent=4)
    logger.info(f"init a empty buffer, and save as {cfg.selector.buffer_file}")


def main():

    torch.manual_seed(3407)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(3407)
    random.seed(3407)

    args = parse_args()


    cfg = Config.fromfile(args.config)
    cfg.local_rank = args.local_rank
    if args.budget is not None:
        cfg.selector.budget = args.budget

    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        cfg.gpus = torch.distributed.get_world_size()
    else:
        cfg.gpus = args.gpus

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info("Distributed testing: {}".format(distributed))
    logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")
    logger.info(f"work dir: {args.work_dir}")

    start_flag = not os.path.exists(cfg.selector.buffer_file)
    if start_flag:
        # init active learning buffer(random sampling)
        init_sample_dataset(cfg, logger)
    else:
        logger.info("using model to predict active learning")
        # init model
        logger.info("init model")
        model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        # init dataset NOTE: use the pipelin of validation, but load the training data
        logger.info("init dataset")
        cfg.data.val.info_path = cfg.selector.infos_origin
        dataset = build_dataset(cfg.data.val)
        data_loader = build_dataloader(
            dataset,
            batch_size=cfg.data.samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
        )
        # load checkpoint
        logger.info("load checkpoint")
        if args.checkpoint is not None:
            checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

        # put model on gpus
        if distributed:
            model = DistributedDataParallel(
                model.cuda(cfg.local_rank),
                device_ids=[cfg.local_rank],
                output_device=cfg.local_rank,
                # broadcast_buffers=False,
                find_unused_parameters=False,
            )
        else:
            model = model.cuda()

        model.eval()

        # init selector
        logger.info("initialize selector")
        selector_cfg = cfg.selector
        selector_cfg.update({
            "detector": model,
            "dataloader": data_loader,
            "logger": logger,
            "pred": args.pred})
        selector: BaseSelector = build_selector(selector_cfg)

        # select samples by selector
        logger.info("begin selection")
        selector.select_samples(local_rank=args.local_rank)
        selector.dump_file()


if __name__ == "__main__":
    main()
