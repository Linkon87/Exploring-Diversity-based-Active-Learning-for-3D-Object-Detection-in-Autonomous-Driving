import logging
import os
from typing import Dict, List, Optional

import torch

from det3d.torchie.trainer import master_only
from det3d.torchie.fileio import load, dump
from .registry import SELECTORS


@SELECTORS.register_module
class BaseSelector(object):
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
        """
        Please fill this initialization in your case
        """
        super().__init__()
        self.budget = budget
        self.buffer_file = buffer_file
        self.dump_file_name = buffer_file if dump_file_name is None else dump_file_name
        self.buffer = load(buffer_file)
        self.detector = detector
        self.dataloader = dataloader
        self.selected_index = {}  # Must be a index list in a dict
        self.infos_file = infos_origin
        self.infos_origin = load(infos_origin)
        self.current_budget = str(self.budget + int(self.get_max_key()))
        self.logger = logger if logger is not None else logging.getLogger(__file__)
        self.pred = pred
        self.cost_b = cost_b
        self.cost_f = cost_f
        self.budget = budget

    def get_max_key(self):
        keys = [int(key) for key in self.buffer.keys()]
        return str(max(keys))

    def select_samples(self, **kwargs) -> None:
        """
        This function performs the selection, please implement it in your case
        """
        return

    @master_only
    def dump_file(self) -> None:
        """
        This function dumps the selected dict to the json file
        You can reimplement this function in your case
        """
        # update buffer
        self.buffer.update(self.selected_index)
        dump(self.buffer, self.dump_file_name)
        self.logger.info(f"update the buffer, and save as {self.dump_file_name}")

        # write infos file
        ext = os.path.splitext(self.infos_file)[-1]
        replace_path = self.infos_file.replace(ext, f"_{self.current_budget}{ext}")
        infos_sampled = [self.infos_origin[i] for i in self.buffer[str(self.current_budget)]]
        dump(infos_sampled, replace_path)
        self.logger.info(f"sample the {self.current_budget} infos and save as {replace_path}")

    def get_selected_samples(self):
        return self.selected_index

    def get_cost_amount(self) -> None:
        cost = 0
        sampled_frames = [self.infos_origin[i] for i in self.buffer[self.get_max_key()]]
        # frames
        cost += self.cost_f * len(sampled_frames)
        # boxes
        for anno in sampled_frames:
            cost += anno["gt_names"].shape[0] * self.cost_b
        return cost
