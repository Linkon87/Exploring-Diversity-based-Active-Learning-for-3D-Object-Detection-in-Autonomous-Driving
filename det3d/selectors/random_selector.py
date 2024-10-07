import random
import logging
import sys
from typing import Dict, List, Optional
import torch

from .base_selector import BaseSelector
from .registry import SELECTORS


@SELECTORS.register_module
class RandomSelector(BaseSelector):
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
        # get the left index to sample

        sampled_index_list = self.buffer[self.get_max_key()]
        origin_index_list = list(range(len(self.infos_origin)))
        left_index_list = origin_index_list.copy()
        for x in sampled_index_list:
            left_index_list.remove(x)

        cost_amount = self.get_cost_amount()
        self.logger.info(f"have selected {self.get_max_key()} examples, "
                         f"need to sample {self.budget} examples from "
                         f"{len(left_index_list)} examples")

        # sample the left index and save
        selected_index_list = []
        while True:
            selected_index = random.choice(left_index_list)
            cost_amount += self.cost_f
            cost_amount += self.infos_origin[selected_index]["gt_names"].shape[0] * self.cost_b
            if cost_amount > int(self.current_budget):
                break
            sys.stdout.write(f"cost amount: ({cost_amount}/{self.current_budget})\r")
            sys.stdout.flush()
            selected_index_list.append(selected_index)
            left_index_list.remove(selected_index)

        self.selected_index[self.current_budget] = selected_index_list + sampled_index_list
