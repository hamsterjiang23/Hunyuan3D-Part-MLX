"""Coarse semantic part splitting for 3D meshes."""

from .contracts import PartRecord, SplitManifest
from .pipeline import inspect_asset, split_asset

__all__ = ["PartRecord", "SplitManifest", "inspect_asset", "split_asset"]
__version__ = "0.1.0"
