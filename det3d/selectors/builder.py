from det3d.utils import build_from_cfg

from .registry import (
    SELECTORS
)


def build_selector(cfg):
    return build_from_cfg(cfg, SELECTORS)
