from .base_selector import BaseSelector
from .random_selector import RandomSelector
from .spatial_selector import SpatialSelector
from .temporal_selector import TemporalSelector
from .feature_selector import FeatureSelector
from .entropy_selector import EntropySelector
from .euclidean_spatial_selector import EuSpatialSelector
"""multi-modality fusion"""
# spatial+temporal(linear/exp)
from .spatial_temporal_selector import SpatialTemporalSelector
from .spatial_temporal_feature_selector import SpatialTemporalFeatureSelector
from .spatial_feature_selector import SpatialFeatureSelector
from .badge_selector import BadgeSelector
from .ppal_selector import PPALSelector
from .cald_selector import CaldSelector
from  .uwe_selector import UWESelector
from .registry import SELECTORS
from .builder import build_selector

__all__ = ["BaseSelector", "RandomSelector", "SpatialSelector", "EuSpatialSelector",
           "TemporalSelector", "SpatialTemporalSelector",
           "SpatialFeatureSelector", "SpatialTemporalFeatureSelector",
           "BadgeSelector",'PPALSelector','CaldSelector','UWESelector',
           "SELECTORS", "build_selector"]
