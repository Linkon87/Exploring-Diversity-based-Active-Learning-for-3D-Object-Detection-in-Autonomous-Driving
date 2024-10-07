import random
import sys
import logging
from typing import Dict, List, Optional
import torch

from det3d import torchie
from det3d.torchie.apis.train import example_to_device
from .base_selector import BaseSelector
from .registry import SELECTORS


@SELECTORS.register_module
class EntropySelector(BaseSelector):
    def __init__(
            self,
            budget: int,
            buffer_file: str,
            dump_file_name: Optional[str] = None,
            infos_origin: List[Dict] = [],
            buffer_path: str = "/home/st2000/data/buffers/entropy_pred.pt",
            p: int = 2,
            detector: Optional[torch.nn.Module] = None,
            dataloader: Optional[torch.utils.data.DataLoader] = None,
            logger: Optional[logging.Logger] = None,
            pred: bool = True,
            random_sample: bool = False,
            sample_num: int = 6000,
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
        assert p in [1, 2]
        self.p = p
        self.random_sample = random_sample
        self.sample_num = sample_num

    def buffer_pred(self, **kwargs) -> torch.Tensor:
        self.logger.info(
            f"begin predict all results of samples and save them as {self.buffer_path}")
        # define the cpu device to release gpu memory
        cpu_device = torch.device("cpu")
        prog_bar = torchie.ProgressBar(len(self.dataloader.dataset))
        b_id = 0
        entropy_list = []
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

                for b_i, pred in enumerate(preds):
                    scores = pred["scores"]
                    entropy = -scores * torch.log(scores) - (1.0 - scores) * torch.log(1 - scores)
                    entropy_list.append(entropy.mean())
                    prog_bar.update()
                    b_id += 1

        entropy = torch.stack(entropy_list)
        torch.save(entropy, self.buffer_path)

        del self.detector
        del fpn_feats
        del pillar_feats

        return entropy

    def select_samples(self, **kwargs) -> None:
        # get the left index
        origin_index_list = list(range(len(self.infos_origin)))
        sampled_index_list = self.buffer[self.get_max_key()]
        left_index_list = origin_index_list.copy()
        for x in sampled_index_list:
            left_index_list.remove(x)

        if self.pred:
            entropy = self.buffer_pred(**kwargs)
            torch.cuda.empty_cache()
            # save the entropy results
            self.logger.info(f"all entropy results have been saved in {self.buffer_path}")
        else:
            # load buffer
            entropy = torch.load(self.buffer_path)
            # load the entropy results
            self.logger.info(f"all entropy results have been load from {self.buffer_path}")

        self.logger.info(f"have selected {self.get_max_key()} examples, "
                         f"need to sample {self.budget} examples from "
                         f"{len(left_index_list)} examples")

        self.logger.info(f"begin to calculate distances map")

        self.logger.info("get the distance map")
        sampled_index_list = self.buffer[self.get_max_key()]

        self.logger.info(f"all prediction results have been load from {self.buffer_path}")

        if self.random_sample:
            assert self.sample_num > 0
            left_index_list = random.sample(left_index_list, self.sample_num)
            self.logger.info(f"use the random sample trick, and randomly "
                             f"sample {self.sample_num} samples in advance.")

        entropy = entropy[left_index_list]  # .cpu().numpy()

        sorted_preds_index = torch.argsort(-entropy).cpu().numpy().tolist()
        selected_index_list = [left_index_list[sorted_preds_index[0]]]

        # sample the left index and save
        cost_amount = self.get_cost_amount()
        cost_amount += self.cost_f
        cost_amount += self.infos_origin[sorted_preds_index[0]]["gt_names"].shape[0] * self.cost_b
        sort_id = 1
        while True:
            selected_index = left_index_list[sorted_preds_index[sort_id]]
            sort_id += 1
            assert selected_index not in selected_index_list, f"id: {selected_index} has been selected"

            cost_amount += self.cost_f
            cost_amount += self.infos_origin[selected_index]["gt_names"].shape[0] * self.cost_b
            if cost_amount > int(self.current_budget):
                break
            sys.stdout.write(f"cost amount: ({cost_amount}/{self.current_budget})\r")
            sys.stdout.flush()
            selected_index_list.append(selected_index)

        self.selected_index[self.current_budget] = selected_index_list + sampled_index_list
