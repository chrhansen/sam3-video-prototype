#!/usr/bin/env python3
"""RunPod pod helper for SAM3 Video prototype."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_BASE = "https://rest.runpod.io/v1"
LAST_POD_ID_PATH = Path(__file__).resolve().parents[1] / ".runpod_pod_id"
DEFAULT_IMAGE = "ghcr.io/chrhansen/sam3-video-prototype:latest"
DEFAULT_GPU_TYPES = [
    "NVIDIA A40",
    "NVIDIA RTX A6000",
    "NVIDIA GeForce RTX 4090",
]
ENV_FILES = [
    Path(__file__).resolve().parents[1] / ".env",
    Path("/Users/chrh/dev/poser/.env"),
]


def load_env_files() -> None:
    for path in ENV_FILES:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().removeprefix("export ").strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_env_pairs(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in values:
        if "=" not in pair:
            raise SystemExit(f"Invalid --env value '{pair}'. Expected KEY=VALUE")
        key, value = pair.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"Invalid --env key in '{pair}'")
        out[key] = value
    return out


def require_api_key() -> str:
    api_key = os.getenv("RUNPOD_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("RUNPOD_API_KEY missing. Set it in env or /Users/chrh/dev/poser/.env")
    return api_key


def require_hf_token(cli_value: str | None) -> str:
    token = (cli_value or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or "").strip()
    if not token:
        raise SystemExit("HF_TOKEN/HUGGINGFACE_TOKEN missing. Needed for gated facebook/sam3 checkpoint access")
    return token


def save_last_pod_id(pod_id: str) -> None:
    LAST_POD_ID_PATH.write_text(f"{pod_id}\n", encoding="utf-8")


def load_last_pod_id() -> str:
    if not LAST_POD_ID_PATH.exists():
        raise SystemExit("No stored pod id. Pass --pod-id or run create first.")
    pod_id = LAST_POD_ID_PATH.read_text(encoding="utf-8").strip()
    if not pod_id:
        raise SystemExit("Stored pod id file empty")
    return pod_id


def resolve_pod_id(value: str | None) -> str:
    return value if value else load_last_pod_id()


def proxy_url(pod_id: str, port: int) -> str:
    return f"https://{pod_id}-{port}.proxy.runpod.net"


def runpod_request(
    method: str,
    path: str,
    *,
    api_key: str,
    payload: dict[str, Any] | None = None,
    query: dict[str, Any] | None = None,
    timeout_s: int = 120,
) -> Any:
    url = f"{API_BASE}{path}"
    if query:
        clean_query: dict[str, str] = {}
        for key, value in query.items():
            if isinstance(value, bool):
                clean_query[key] = "true" if value else "false"
            else:
                clean_query[key] = str(value)
        url = f"{url}?{urlencode(clean_query)}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = Request(url=url, data=body, method=method, headers=headers)
    try:
        with urlopen(req, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else None
    except HTTPError as err:
        body_text = err.read().decode("utf-8", errors="replace").strip() or "<empty body>"
        raise SystemExit(f"RunPod API {method} {path} failed ({err.code}): {body_text}") from err
    except URLError as err:
        raise SystemExit(f"RunPod API {method} {path} failed: {err}") from err


def get_pod(pod_id: str) -> dict[str, Any]:
    pod = runpod_request(
        "GET",
        f"/pods/{pod_id}",
        api_key=require_api_key(),
        query={"includeMachine": True, "includeNetworkVolume": True},
    )
    if not isinstance(pod, dict):
        raise SystemExit(f"Unexpected pod response: {pod}")
    return pod


def summarize_pod(pod: dict[str, Any], http_port: int) -> str:
    pod_id = str(pod.get("id", "<unknown>"))
    name = str(pod.get("name", "<unknown>"))
    desired_status = str(pod.get("desiredStatus", "<unknown>"))
    actual_status = str(pod.get("runtime", {}).get("uptimeInSeconds", "n/a"))
    image = str(pod.get("imageName") or pod.get("image") or "<unknown>")
    return "\n".join(
        [
            f"pod_id: {pod_id}",
            f"name: {name}",
            f"desired_status: {desired_status}",
            f"uptime_seconds: {actual_status}",
            f"image: {image}",
            f"proxy_url: {proxy_url(pod_id, http_port)}",
        ]
    )


def create_command(args: argparse.Namespace) -> None:
    api_key = require_api_key()
    hf_token = require_hf_token(args.hf_token)
    gpu_type_ids = split_csv(args.gpu_types)
    if not gpu_type_ids:
        raise SystemExit("At least one GPU type required")

    runtime_env = {
        "PORT": str(args.http_port),
        "HF_TOKEN": hf_token,
        "HUGGINGFACE_TOKEN": hf_token,
        "SAM3_DEVICE": "cuda",
        "SAM3_LOAD_FROM_HF": "1",
        "HF_HOME": "/workspace/.cache/huggingface",
        "HUGGINGFACE_HUB_CACHE": "/workspace/.cache/huggingface/hub",
    }
    runtime_env.update(parse_env_pairs(args.env))

    payload: dict[str, Any] = {
        "name": args.name or f"sam3-video-{int(time.time())}",
        "cloudType": args.cloud_type,
        "computeType": "GPU",
        "imageName": args.image,
        "gpuCount": 1,
        "gpuTypeIds": gpu_type_ids,
        "containerDiskInGb": args.container_disk_gb,
        "volumeInGb": args.volume_gb,
        "volumeMountPath": args.volume_mount_path,
        "ports": [f"{args.http_port}/http"],
        "interruptible": args.interruptible,
        "env": runtime_env,
    }
    if args.bootstrap_script_url:
        start_cmd = (
            "set -euo pipefail; "
            "mkdir -p /workspace; "
            f"python3 -c \"from urllib.request import urlopen; "
            f"open('/tmp/bootstrap-sam3-video.sh', 'wb').write(urlopen({args.bootstrap_script_url!r}, timeout=300).read())\"; "
            "chmod +x /tmp/bootstrap-sam3-video.sh; "
            f"bash /tmp/bootstrap-sam3-video.sh {shlex.quote(args.app_dir)}"
        )
        payload["dockerStartCmd"] = ["bash", "-lc", start_cmd]
    if args.data_centers:
        payload["dataCenterIds"] = split_csv(args.data_centers)

    pod = runpod_request("POST", "/pods", api_key=api_key, payload=payload)
    pod_id = str(pod.get("id", "")).strip()
    if not pod_id:
        raise SystemExit(f"Unexpected create response: {json.dumps(pod, indent=2)}")

    save_last_pod_id(pod_id)
    print("Created pod")
    print(summarize_pod(pod, args.http_port))
    print(f"stored_pod_id_file: {LAST_POD_ID_PATH}")

    if args.wait:
        wait_for_command(
            argparse.Namespace(
                pod_id=pod_id,
                timeout_s=args.wait_timeout_s,
                interval_s=args.wait_interval_s,
                http_port=args.http_port,
            )
        )


def status_command(args: argparse.Namespace) -> None:
    pod = get_pod(resolve_pod_id(args.pod_id))
    print(summarize_pod(pod, args.http_port))


def wait_for_command(args: argparse.Namespace) -> None:
    pod_id = resolve_pod_id(args.pod_id)
    deadline = time.time() + args.timeout_s
    ready = proxy_url(pod_id, args.http_port) + "/readyz"

    while time.time() < deadline:
        try:
            pod = get_pod(pod_id)
            with urlopen(Request(ready, headers={"User-Agent": "curl/8.5.0"}), timeout=10) as response:
                if response.status == 200:
                    print("Pod ready")
                    print(summarize_pod(pod, args.http_port))
                    return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(args.interval_s)

    raise SystemExit(f"Timed out waiting for {ready}")


def pod_action_command(args: argparse.Namespace) -> None:
    pod_id = resolve_pod_id(args.pod_id)
    pod = runpod_request("POST", f"/pods/{pod_id}/{args.action}", api_key=require_api_key(), payload={})
    print(f"{args.action}: {pod_id}")
    if isinstance(pod, dict):
        print(summarize_pod(pod, args.http_port))


def delete_command(args: argparse.Namespace) -> None:
    pod_id = resolve_pod_id(args.pod_id)
    runpod_request("DELETE", f"/pods/{pod_id}", api_key=require_api_key())
    if LAST_POD_ID_PATH.exists() and LAST_POD_ID_PATH.read_text(encoding="utf-8").strip() == pod_id:
        LAST_POD_ID_PATH.unlink()
    print(f"deleted: {pod_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RunPod helper for SAM3 Video prototype")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    create.add_argument("--image", default=DEFAULT_IMAGE)
    create.add_argument("--name", default="")
    create.add_argument("--hf-token", default="")
    create.add_argument("--gpu-types", default=",".join(DEFAULT_GPU_TYPES))
    create.add_argument("--cloud-type", default="SECURE")
    create.add_argument("--data-centers", default=os.getenv("RUNPOD_DATA_CENTERS", "EU-SE-1"))
    create.add_argument("--http-port", type=int, default=8000)
    create.add_argument("--container-disk-gb", type=int, default=80)
    create.add_argument("--volume-gb", type=int, default=40)
    create.add_argument("--volume-mount-path", default="/workspace")
    create.add_argument("--bootstrap-script-url", default="")
    create.add_argument("--app-dir", default="/workspace/sam3-video")
    create.add_argument("--interruptible", action="store_true")
    create.add_argument("--env", action="append", default=[])
    create.add_argument("--wait", action="store_true")
    create.add_argument("--wait-timeout-s", type=int, default=1800)
    create.add_argument("--wait-interval-s", type=int, default=10)
    create.set_defaults(func=create_command)

    status = subparsers.add_parser("status")
    status.add_argument("--pod-id")
    status.add_argument("--http-port", type=int, default=8000)
    status.set_defaults(func=status_command)

    wait = subparsers.add_parser("wait")
    wait.add_argument("--pod-id")
    wait.add_argument("--http-port", type=int, default=8000)
    wait.add_argument("--timeout-s", type=int, default=1800)
    wait.add_argument("--interval-s", type=int, default=10)
    wait.set_defaults(func=wait_for_command)

    for action in ("start", "stop", "restart"):
        sub = subparsers.add_parser(action)
        sub.add_argument("--pod-id")
        sub.add_argument("--http-port", type=int, default=8000)
        sub.set_defaults(func=pod_action_command, action=action)

    delete = subparsers.add_parser("delete")
    delete.add_argument("--pod-id")
    delete.set_defaults(func=delete_command)

    return parser


def main() -> None:
    load_env_files()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
