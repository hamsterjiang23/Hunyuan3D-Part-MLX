from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn

from split3d.hunyuan.service import MLXPartRunner, PartJobService, create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Hunyuan3D-Part through a local async MLX worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument("--cache-path", type=Path, default=Path("server_cache"))
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--p3-weights", type=Path, required=True)
    parser.add_argument("--retention-seconds", type=int, default=86_400)
    parser.add_argument("--max-mesh-mib", type=int, default=100)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    runner = MLXPartRunner(args.model_dir, args.p3_weights)
    service = PartJobService(
        args.cache_path,
        args.input_root,
        runner,
        retention_seconds=args.retention_seconds,
        max_mesh_bytes=args.max_mesh_mib * 1024 * 1024,
    )
    service.cleanup()
    try:
        uvicorn.run(create_app(service), host=args.host, port=args.port, log_level=args.log_level)
    finally:
        service.shutdown()


if __name__ == "__main__":
    main()
