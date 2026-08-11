from __future__ import annotations

import argparse
import json
from pathlib import Path

from split3d.hunyuan.weights import inspect_safetensors


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect and validate safetensors manifests without loading weights")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--match", help="only include tensor names containing this value")
    parser.add_argument("--rank", type=int, help="only include tensors with this number of dimensions")
    args = parser.parse_args()

    results = []
    for path in args.paths:
        manifest = inspect_safetensors(path, require_complete=not args.allow_partial)
        payload = manifest.summary()
        if args.match or args.rank is not None:
            payload["matching_tensors"] = [
                {
                    "name": tensor.name,
                    "dtype": tensor.dtype,
                    "shape": tensor.shape,
                    "data_offsets": tensor.data_offsets,
                }
                for tensor in manifest.tensors
                if (args.match is None or args.match in tensor.name)
                and (args.rank is None or len(tensor.shape) == args.rank)
            ]
        results.append(payload)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
