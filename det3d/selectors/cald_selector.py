import random
import sys
import logging
from typing import Dict, List, Optional
import torch

from det3d import torchie
from det3d.torchie.apis.train import example_to_device
from .base_selector import BaseSelector
from .registry import SELECTORS
import json
import pickle
import collections

# run tools/cald_pred_list.py & tools/cald_ent.py to get  idx_to_jsdiv.pkl   first

@SELECTORS.register_module
class CaldSelector(BaseSelector):
    def __init__(
            self,
            budget: int,
            buffer_file: str,
            dump_file_name: Optional[str] = None,
            infos_origin: List[Dict] = [],
            buffer_path: str = "/home/st2000/data/buffers/cald_ent_sorted_idx.json",
            detector: Optional[torch.nn.Module] = None,
            dataloader: Optional[torch.utils.data.DataLoader] = None,
            logger: Optional[logging.Logger] = None,
            cost_b: float = 0.04,
            cost_f: float = 0.12,
            pred: bool = False
    ) -> None:
        super().__init__(
            budget,
            buffer_file,
            dump_file_name,
            infos_origin=infos_origin,
            detector=detector,
            dataloader=dataloader,
            logger=logger,
            cost_b=cost_b,
            cost_f=cost_f,
            pred = pred
        )
        self.buffer_path = buffer_path


    def select_samples(self, **kwargs) -> None:
        # get the left index
        origin_index_list = list(range(len(self.infos_origin)))
        sampled_index_list = self.buffer[self.get_max_key()]
        left_index_list = origin_index_list.copy()
        for x in sampled_index_list:
            left_index_list.remove(x)

        with open(self.buffer_path) as f:
            entropy = json.load(f)
            # load the entropy results
        self.logger.info(f"all entropy results have been load from {self.buffer_path}")

        self.logger.info(f"have selected {self.get_max_key()} examples, "
                         f"need to sample {self.budget} examples from "
                         f"{len(left_index_list)} examples")


        sampled_index_list = self.buffer[self.get_max_key()]

        self.logger.info(f"all prediction results have been load from {self.buffer_path}")

        for x in sampled_index_list:
            entropy.remove(x)
        # entropy = entropy[left_index_list]  # .cpu().numpy()

        sorted_preds_index = entropy #torch.argsort(-entropy).cpu().numpy().tolist()
        selected_index_list_ent = [sorted_preds_index[0]]

        # sample the left index and save
        cost_amount = self.get_cost_amount()
        cost_amount += self.cost_f
        cost_amount += self.infos_origin[sorted_preds_index[0]]["gt_names"].shape[0] * self.cost_b
        sort_id = 1
        while True:
            selected_index = sorted_preds_index[sort_id]
            sort_id += 1
            assert selected_index not in selected_index_list_ent, f"id: {selected_index} has been selected"

            cost_amount += self.cost_f
            cost_amount += self.infos_origin[selected_index]["gt_names"].shape[0] * self.cost_b
            if cost_amount > int(self.current_budget) + self.budget * 0.5 :
                break
            selected_index_list_ent.append(selected_index)

        # self.selected_index[self.current_budget] = selected_index_list_ent + sampled_index_list

        ## hard code
        with open('/home/linjp/share/ActiveLearn4Detection-main/idx_to_jsdiv.pkl','rb') as f:
            idx_to_jsdiv = pickle.load(f)

        sorted_idx_to_jsdiv = sorted(idx_to_jsdiv.items(), key=lambda x: x[1], reverse=True)
        sorted_dict = collections.OrderedDict(sorted_idx_to_jsdiv)

        sort_to_select = list(sorted_dict.keys())
        for i in sort_to_select:
            if i in selected_index_list_ent:
                idx = i
                selected_index_list = [idx]
                sort_to_select.remove(i)
                break
            sort_to_select.remove(i)

        # sample the left index and save
        left_index_list = selected_index_list_ent
        cost_amount = self.get_cost_amount()
        cost_amount += self.cost_f
        cost_amount += self.infos_origin[idx]["gt_names"].shape[0] * self.cost_b
        while True:

            for i in sort_to_select:
                if i in selected_index_list_ent:
                    selected_index = i
                    sort_to_select.remove(i)
                    break
                sort_to_select.remove(i)

            assert selected_index not in selected_index_list, f"id: {selected_index} has been selected"

            cost_amount += self.cost_f
            cost_amount += self.infos_origin[selected_index]["gt_names"].shape[0] * self.cost_b
            if cost_amount > int(self.current_budget):
                break
            sys.stdout.write(f"cost amount: ({cost_amount}/{self.current_budget})\r")
            sys.stdout.flush()
            selected_index_list.append(selected_index)

        print('')
        self.selected_index[self.current_budget] = selected_index_list + sampled_index_list

