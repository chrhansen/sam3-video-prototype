# SAM3 Video skier track prototype

Upload short ski clip. Choose exactly one prompt mode:

- click frame 0
- text prompt like `person` or `skier in orange jacket and black pants`

Output:

- `overlay.mp4`
- `masks.json` (`masktrack.v1`)

Mask overlay matches the earlier SAM2 / SAM3 tracker prototypes:

- green segmentation tint
- thin black boundary outline on every frame

## Model choice

This uses official Meta SAM3 Video, not SAM3 Tracker.

Why:

- same predictor API supports dense text prompting
- same predictor also supports point-based instance interactivity on video
- one backend can serve both requested prompt modes

Dependency pinned to official repo commit:

- `facebookresearch/sam3@86ed77094094e5cabb16b0414ec60c5ba9ce0a0f`

## Behavior

- Click mode: single target, object id `1`.
- Text mode: SAM3 Video may detect multiple matches. This prototype keeps the top-scoring match and renders only that target.
- Each upload job uses a fresh predictor instance. Repeated uploads on the same pod are supported.
- Outputs stored in `runs/<job_id>/`.

## Local quickstart

CUDA GPU strongly recommended. Official predictor path here expects CUDA.

1. Create env:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install PyTorch + torchvision for CUDA 12.6:

```bash
pip install --upgrade pip "setuptools<82" wheel
pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.7.0 torchvision==0.22.0
```

3. Install app deps:

```bash
pip install -r requirements.txt
```

4. Authenticate for gated `facebook/sam3` access:

```bash
export HF_TOKEN=...
```

5. Run:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

## Docker / RunPod image

Build local image:

```bash
docker buildx build --platform linux/amd64 -t sam3-video:local -f Dockerfile.runpod .
```

Run local GPU container:

```bash
docker run --rm --gpus all -p 8000:8000 \
  -e HF_TOKEN="$HF_TOKEN" \
  sam3-video:local
```

Published GHCR image:

```bash
ghcr.io/chrhansen/sam3-video-prototype:latest
```

## RunPod pod helper

Create pod from the published GHCR image:

```bash
python scripts/runpod_pod.py create --wait
```

Check pod:

```bash
python scripts/runpod_pod.py status
python scripts/runpod_pod.py wait
```

Stop / start / restart / delete:

```bash
python scripts/runpod_pod.py stop
python scripts/runpod_pod.py start
python scripts/runpod_pod.py restart
python scripts/runpod_pod.py delete
```

The helper auto-loads:

- repo `.env` if present
- `/Users/chrh/dev/poser/.env`

Required runtime secrets:

- `RUNPOD_API_KEY`
- `HF_TOKEN`

## Bootstrap fallback

If GHCR image pull stalls on a RunPod host, use a stock GPU image and bootstrap from a repo checkout. This is the fallback that produced the current working RunPod pod.

Recommended stock image:

```bash
pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel
```

Inside the pod:

```bash
git clone https://github.com/chrhansen/sam3-video-prototype.git /workspace/sam3-video
cd /workspace/sam3-video
bash scripts/bootstrap_runpod.sh /workspace/sam3-video
```

Notes:

- `scripts/bootstrap_runpod.sh` installs app deps plus `torchvision==0.22.0` if the base image does not already provide it.
- The script assumes the base image already ships with a compatible CUDA `torch` build.
- If you need a fixed deploy, `git checkout <sha>` before running the bootstrap script.

## GitHub Actions / GHCR

Two workflows ship in `.github/workflows/`:

- `ci.yml`: compile + unit test smoke gate
- `docker-publish.yml`: builds `Dockerfile.runpod`, pushes `ghcr.io/chrhansen/sam3-video-prototype`

Published tags:

- `latest` on default branch
- `sha-<shortsha>` on each push

## Endpoints

- `/`
- `/readyz`
- `/healthz`
- `/upload`
- `/prompt/{job_id}`
- `/status/{job_id}`

## Notes

- Official SAM3 Video predictor loads large weights. First prompt on a cold pod will take time.
- `SAM3_COMPILE=1` exists but left off by default. Compile cost is not prototype-friendly.
- If RunPod image pull or gated checkpoint download fails, verify `HF_TOKEN` scope/access first.
- Some RunPod hosts have stalled on the GHCR app image; the stock-image bootstrap flow above is the operational fallback.
- If `SAM3_LOAD_FROM_HF=0`, also set `SAM3_CHECKPOINT_PATH` to a local checkpoint file.
