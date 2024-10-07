# shm at least 64G
import argparse
import json
import os
import pickle
import sys

import cv2
import numpy as np
import torch
import yaml
from det3d import __version__, torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.datasets.kitti import kitti_common as kitti
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    batch_processor,
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)
from det3d.torchie.trainer import get_dist_info, load_checkpoint
from det3d.torchie.trainer.utils import all_gather, synchronize
from torch.nn.parallel import DistributedDataParallel

from tools.demo_utils import visual
import pickle
from torch.utils.data import DataLoader
from det3d.torchie.parallel import collate, collate_kitti


def convert_box(info):
    boxes = info["gt_boxes"].astype(np.float32)
    names = info["gt_names"]

    assert len(boxes) == len(names)

    detection = {}

    detection['box3d_lidar'] = boxes

    # dummy value
    detection['label_preds'] = np.zeros(len(boxes))
    detection['scores'] = np.ones(len(boxes))

    return detection


def save_pred(pred, root):
    with open(os.path.join(root, "prediction.pkl"), "wb") as f:
        pickle.dump(pred, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Test a detector")
    parser.add_argument("--config", help="train config file path",
                        default="/home/linjp/share/ActiveLearn4Detection-main/examples/active/cbgs_ppal.py",
                        )
    parser.add_argument("--work_dir", help="the dir to save logs and models",
                        default="/home/st2000/data/work_dir/seed42/badge/NUSC_CBGS_badge-debug"
                        )
    parser.add_argument(
        "--checkpoint", help="the dir to checkpoint which the model read from",
       default="/home/st2000/data/work_dir/seed42/NUSC_CBGS_random_600_20221127-184253/latest.pth"
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
        default=1,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="pytorch",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main():
    # torch.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(0)
    torch.multiprocessing.set_sharing_strategy('file_system')

    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.local_rank = args.local_rank

    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)

    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

        cfg.gpus = torch.distributed.get_world_size()
    else:
        cfg.gpus = args.gpus

    logger.info("Distributed testing: {}".format(distributed))
    logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        batch_size=1,#cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        num_gpus=1,
        dist=distributed,
        shuffle=False,
    )
    # data_loader = DataLoader(
    #     dataset,
    #     batch_size=1,
    #     sampler=None,
    #     shuffle=False,
    #     num_workers=1,
    #     collate_fn=collate_kitti,
    #     pin_memory=False,
    # )

    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

    # put model on gpus
    if distributed:
        model = DistributedDataParallel(
            model.cuda(cfg.local_rank),
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank,
            # broadcast_buffers=False,
            find_unused_parameters=True,
        )
    else:
        model = model.cuda()

    model.eval()
    mode = "val"

    logger.info(f"work dir: {args.work_dir}")

    if cfg.local_rank == 0:
        prog_bar = torchie.ProgressBar(len(data_loader.dataset) // cfg.gpus)

    detections = {}
    cpu_device = torch.device("cpu")
    points_list = []
    gt_annos = []
    detections2 = []
    sample_token = []

    for i, data_batch in enumerate(data_loader):
        # if i % 1000 != 0:continue
        with torch.no_grad():
            outputs = batch_processor(
                model, data_batch, train_mode=False, local_rank=args.local_rank,
            )
        # info = dataset._nusc_infos[i]
        # gt_annos.append(convert_box(info))
        # sample_token.append(dataset._nusc_infos[i]['token'])
        # points = data_batch['points'][:, 1:4].cpu().numpy()
        for output in outputs:
            token = output["metadata"]["token"]
            for k, v in output.items():
                if k not in [
                    "metadata",
                ]:
                    output[k] = v.to(cpu_device)
            detections.update(
                {token: output, }
            )
            # detections2.append(output)
            if args.local_rank == 0:
                prog_bar.update()
        # points_list.append(points.T)

    synchronize()

    # print('Done model inference. Please wait a minute, the matplotlib is a little slow...')
    # print("len(list)",len(points_list),end='\n')
    # import json
    # import pickle
    # with open('tensor_list.pkl', 'wb') as f:
    #     pickle.dump(detections2, f)

    # data_det = detections2
    # with open('detections2.json', 'w', encoding='utf-8') as f:
    #     json.dump(data_det, f, default=str,ensure_ascii=False)

    # for i in range(len(points_list)):
    #     visual(points_list[i], gt_annos[i], sample_token[i], detections2[i], i) ####可视化
    #     print("Rendered Image {}".format(i))

    # image_folder = 'diversity-gt'
    # # video_name = 'diversity.avi'
    # images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    # images.sort(key=lambda img_name: int(img_name.split('.')[0][4:]))
    # frame = cv2.imread(os.path.join(image_folder, images[0]))
    # height, width, layers = frame.shape
    # #video = cv2.VideoWriter(video_name, 0, 1, (width, height))
    # cv2_images = []
    #
    # for image in images:
    #     cv2_images.append(cv2.imread(os.path.join(image_folder, image)))
    #
    # # for img in cv2_images:
    # #     video.write(img)
    # cv2.destroyAllWindows()
    # # video.release()
    # print("Successfully save video in the main folder")

    synchronize()

    all_predictions = all_gather(detections)

    if args.local_rank != 0:
        return

    predictions = {}
    for p in all_predictions:
        predictions.update(p)

    # f_save = open('predictions.pkl', 'wb')
    # pickle.dump(predictions, f_save)
    # f_save.close()

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    # save_pred(predictions, args.work_dir)

    result_dict, _ = dataset.evaluation(predictions, output_dir=args.work_dir)

    for k, v in result_dict["results"].items():
        print(f"Evaluation {k}: {v}")

    # if args.txt_result:
    #     res_dir = os.path.join(os.getcwd(), "predictions")
    #     for k, dt in predictions.items():
    #         with open(
    #                 os.path.join(res_dir, "%06d.txt" % int(dt["metadata"]["token"])), "w"
    #         ) as fout:
    #             lines = kitti.annos_to_kitti_label(dt)
    #             for line in lines:
    #                 fout.write(line + "\n")

        # ap_result_str, ap_dict = kitti_evaluate(
        #     "/data/Datasets/KITTI/Kitti/object/training/label_2",
        #     res_dir,
        #     label_split_file="/data/Datasets/KITTI/Kitti/ImageSets/val.txt",
        #     current_class=0,
        # )
        #
        # print(ap_result_str)


if __name__ == "__main__":
    main()
