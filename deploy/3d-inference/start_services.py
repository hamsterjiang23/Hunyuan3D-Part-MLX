"""
3D Inference Service Launcher

Architecture:
    Queue Proxy (:8080)  ← external-facing, shared queue
        ├── TRELLIS backend (:8081)  — PyTorch MPS
        ├── Hunyuan3D backend (:8082) — MLX
        └── Hunyuan3D-Part backend (:8083) — MLX

Usage:
    python start_services.py start      # Start all services
    python start_services.py stop       # Stop all services
    python start_services.py status     # Check status
    python start_services.py restart    # Restart all services
"""

import argparse

# Deployment snapshot for the Mac service stack; adjust host paths when reused.
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(BASE_DIR, "run")
LOG_DIR = os.path.join(BASE_DIR, "logs")
PID_FILE = os.path.join(RUN_DIR, "pids.json")
LEGACY_PID_FILE = "/tmp/3d-inference-pids.json"

SERVICES = {
    "queue": {
        "port": 8080,
        "host": "0.0.0.0",
        "health_path": "/ready",
        "cwd": BASE_DIR,
        "venv": os.path.join(BASE_DIR, ".venv"),
        "script": "queue_server.py",
        "args": [],  # --host/--port/--cache-path are passed by the launcher
    },
    "trellis": {
        "port": 8081,
        "host": "127.0.0.1",
        "health_path": "/health",
        "cwd": os.path.expanduser("~/hamster/trellis-mac"),
        "venv": os.path.expanduser("~/hamster/trellis-mac/.venv"),
        "script": "api_server.py",
        "args": [],
    },
    "hunyuan3d": {
        "port": 8082,
        "host": "127.0.0.1",
        "health_path": "/health",
        "cwd": os.path.expanduser("~/hamster/Hunyuan3D-2.1-mlx"),
        "venv": os.path.expanduser("~/hamster/Hunyuan3D-2.1-mlx/.venv"),
        "script": "api_server_mlx.py",
        "args": [],
    },
    "hunyuan3d_part": {
        "port": 8083,
        "host": "127.0.0.1",
        "health_path": "/health",
        "cwd": os.path.expanduser("~/hamster/3D_Split"),
        "venv": os.path.expanduser("~/hamster/3D_Split/.venv"),
        "script": "scripts/serve_hunyuan_part_mlx.py",
        "args": [
            "--input-root", os.path.join(BASE_DIR, "server_cache", "inputs"),
            "--model-dir", os.path.expanduser("~/hamster/models/Hunyuan3D-Part"),
            "--p3-weights", os.path.expanduser("~/hamster/3D_Split/models/p3sam.safetensors"),
        ],
    },
    "mesh_mcp": {
        "port": 8090,
        "host": "0.0.0.0",
        "health_path": "/health",
        "cwd": BASE_DIR,
        "venv": os.path.join(BASE_DIR, ".venv"),
        "script": "mesh_service_mcp.py",
        # transport must be http (not stdio); public-url = LAN IP so
        # download links work from other machines
        "args": ["--transport", "http",
                 "--public-url", "http://10.20.134.22:8090"],
    },
}


def _wait_for_health(url: str, timeout: int = 15) -> bool:
    for i in range(timeout):
        try:
            resp = urllib.request.urlopen(url, timeout=2)
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _write_json_atomic(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = f"{path}.tmp"
    with open(temp_path, "w") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_path, path)


def _start_one(name: str, info: dict) -> dict:
    python = os.path.join(info["venv"], "bin", "python")
    script = os.path.join(info["cwd"], info["script"])
    port = info["port"]
    cache = os.path.join(info["cwd"], "server_cache")
    os.makedirs(cache, exist_ok=True)

    cmd = [
        python,
        script,
        "--host",
        info["host"],
        "--port",
        str(port),
        "--cache-path",
        cache,
    ] + info["args"]

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{name}.log")
    with open(log_path, "ab", buffering=0) as log_stream:
        proc = subprocess.Popen(
            cmd,
            cwd=info["cwd"],
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    health_url = f"http://127.0.0.1:{port}{info['health_path']}"
    if not _wait_for_health(health_url, timeout=20) or proc.poll() is not None:
        if proc.poll() is None:
            proc.terminate()
        raise RuntimeError(f"health check failed; see {log_path}")

    print(f"  {name:<12} port {port}  PID {proc.pid}  log {log_path}")

    return {
        "pid": proc.pid,
        "port": port,
        "host": info["host"],
        "log": log_path,
        "started_at": time.time(),
    }


def cmd_start(args):
    occupied = [
        str(info["port"]) for info in SERVICES.values() if _port_is_open(info["port"])
    ]
    if occupied:
        print(f"Cannot start: port(s) already in use: {', '.join(occupied)}")
        print("Run 'python start_services.py restart' to replace existing services.")
        return False

    print("Starting 3D inference services...")
    print(f"  Queue Proxy     → http://localhost:8080  (submit tasks here)")
    print(f"  TRELLIS backend → http://localhost:8081")
    print(f"  Hunyuan3D backend → http://localhost:8082")
    print(f"  Hunyuan3D-Part backend → http://localhost:8083")
    print()

    pids = {}
    for name in ("trellis", "hunyuan3d", "hunyuan3d_part", "queue", "mesh_mcp"):
        try:
            pids[name] = _start_one(name, SERVICES[name])
        except Exception as e:
            print(f"  {name} failed: {e}")
            for started in reversed(list(pids.values())):
                try:
                    os.kill(started["pid"], signal.SIGTERM)
                except ProcessLookupError:
                    pass
            return False

    _write_json_atomic(PID_FILE, pids)

    print()
    print("=" * 60)
    return True


def _read_pid_file() -> dict:
    for path in (PID_FILE, LEGACY_PID_FILE):
        if os.path.exists(path):
            with open(path) as stream:
                return json.load(stream)
    return {}


def _remove_pid_files() -> None:
    for path in (PID_FILE, LEGACY_PID_FILE):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def _pids_on_port(port: int) -> list[int]:
    result = subprocess.run(
        ["lsof", "-nP", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
        capture_output=True,
        text=True,
    )
    return [int(value) for value in result.stdout.split() if value.isdigit()]
    print("Clients send requests to http://<your-ip>:8080")
    print()
    print("  POST /submit    {'model':'trellis','image':'<b64>', ...}")
    print("  GET  /status    /status/{uid}")
    print("  GET  /download  /download/{uid}")
    print("  GET  /queue     (see what's queued)")
    print("=" * 60)


def cmd_stop(args):
    recorded = _read_pid_file()
    targets = {}
    for name, info in SERVICES.items():
        pid = recorded.get(name, {}).get("pid")
        candidates = ([pid] if pid else []) + _pids_on_port(info["port"])
        targets[name] = list(
            dict.fromkeys(candidate for candidate in candidates if candidate)
        )

    for name in reversed(list(SERVICES)):
        for pid in targets[name]:
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"  stopping {name} (PID {pid})")
            except ProcessLookupError:
                pass
            except Exception as e:
                print(f"  error stopping {name}: {e}")

    deadline = time.time() + 15
    while time.time() < deadline and any(
        _port_is_open(info["port"]) for info in SERVICES.values()
    ):
        time.sleep(0.25)
    remaining = [
        str(info["port"]) for info in SERVICES.values() if _port_is_open(info["port"])
    ]
    _remove_pid_files()
    if remaining:
        print(f"Failed to stop service port(s): {', '.join(remaining)}")
        return False
    print("All services stopped.")
    return True


def cmd_status(args):
    all_ok = True
    for name, info in SERVICES.items():
        port = info["port"]
        try:
            resp = urllib.request.urlopen(
                f"http://127.0.0.1:{port}{info['health_path']}", timeout=3
            )
            data = json.loads(resp.read())
            details = []
            for key in (
                "model_loaded",
                "shape_loaded",
                "paint_loaded",
                "p3sam_loaded",
                "xpart_loaded",
                "database",
            ):
                if key in data:
                    details.append(f"{key}={data[key]}")
            print(f"  {name:<12} port {port}  {data['status']} {' '.join(details)}")
        except Exception as e:
            print(f"  {name:<12} port {port}  NOT RUNNING")
            all_ok = False
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="3D Inference Service Launcher")
    parser.add_argument(
        "action",
        nargs="?",
        choices=["start", "stop", "status", "restart"],
        default="status",
        help="Action to perform",
    )
    args = parser.parse_args()

    if args.action == "start":
        cmd_start(args)
    elif args.action == "stop":
        cmd_stop(args)
    elif args.action == "restart":
        if cmd_stop(args):
            time.sleep(1)
            cmd_start(args)
    else:
        cmd_status(args)


if __name__ == "__main__":
    main()
