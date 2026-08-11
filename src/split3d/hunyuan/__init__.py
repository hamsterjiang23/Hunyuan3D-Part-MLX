"""Hunyuan3D-Part model support.

This package contains the native MLX port and backend-independent validation
utilities. MLX imports stay inside the modules that execute model code so the
weight and metric tooling remains usable on CUDA and CPU reference machines.
"""

from .weights import SafetensorManifest, TensorSpec, inspect_safetensors

__all__ = ["SafetensorManifest", "TensorSpec", "inspect_safetensors"]
