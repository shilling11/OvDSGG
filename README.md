# PRISM: One-Stage End-to-End Open-Vocabulary Video Scene Graph Generation via Pointer-based Relational Inference

This code has been taken and modified from [TransVOD](https://github.com/SJTU-LuHe/TransVOD/tree/master) GitHub repository.

## Repository structure
```
OvDSGG/
├── configs/                  # Shell configs for training/eval
├── datasets/                 # Dataset loaders + VidVRD evaluator
├── engine_multi.py           # Multi-frame train/eval loop
├── main.py                   # Entry point + arg parsing
├── models/                   # Model definitions
│   ├── backbone.py           # CLIP ViT backbone with VPT
│   ├── deformable_detr_multi.py
│   └── ops/                  # MultiScaleDeformableAttention CUDA op
├── tools/                    # DDP launchers, visualisation
├── util/                     # Misc utilities + text prompt encoder
├── object_embeddings.pt      # Pre-computed CLIP text embeddings (35 obj + 1 bg)
├── predicate_embeddings.pt   # Pre-computed CLIP text embeddings (132 pred + 1 bg)
├── data/                     # SYMLINK → see "Dataset Preparation"
├── exps/                     # SYMLINK → see "Workspace Setup"
└── Dockerfile
```

## Workspace setup
The repo expects two large directories outside the source tree, accessed via symlinks. This keeps the repo portable across machines and avoids committing data/checkpoints.

### `exps/` - checkpoints, logs, eval outputs

```bash
# Pick a location with ample space
mkdir -p /path/to/exps
ln -s /path/to/exps /path/to/OvDSGG/exps
```

After symlink, training scripts will write to `exps/clip_vitb16/<run_name>/` automatically.

### `data/` - VidVRD dataset
```
mkdir -p /path/to/data
ln -s /path/to/data /path/to/OvDSGG/data
```

Data directory will be populated in next section.

## Dataset preparation
```
data/
└── vidvrd/
    ├── vidvrd_videos_train/        # Raw .mp4 (800 train videos)
    ├── vidvrd_videos_test/         # Raw .mp4 (200 test videos)
    ├── vidvrd_extracted_frames/
    │   ├── train/<video_id>/0001.jpg, 0002.jpg, ...
    │   └── val/<video_id>/0001.jpg, ...
    ├── annotations/
    │   ├── instances_train_base.json    # COCO-format annotations (train, base classes only)
    │   ├── instances_val.json           # COCO-format annotations (val, all classes)
    │   ├── train_anno/<video_id>.json   # Original VidVRD per-video annotations
    │   └── test_anno/<video_id>.json
    └── extract_frames.py
```

Step 1:

Download videos from [official source](https://xdshang.github.io/docs/imagenet-vidvrd.html) and place raw `.mp4` files into `data/vidvrd/vidvrd_videos_train/` and `data/vidvrd/vidvrd_videos_test/`.

Step 2:

Extract frames from videos. In `extract_frames.py`, change the `TRAIN_MP4_DIR`, `EVAL_MP4_DIR`, and `OUTPUT_ROOT` variables to match local folder paths, then run file.
```
cd data/vidvrd
python extract_frames.py
```

Step 3:

Convert annotations to COCO format.
```
cd /path/to/OvDSGG/datasets/vidvrd_dataset
python vrd2coco.py
```

### CLIP text embeddings
Repository ships with pre-computed CLIP ViT-B/16 text embeddings.
- `object_embeddings.pt` - shape `[36,512]` (1 background + 35 object classes)
`predicate_embeddings.pt` - shape `[133, 512]` (1 background + 132 predicate classes)

These are used directly by model: no regeneration needed unless you change CLIP variants.

## Environment setup
To run this code for training and evaluation, it is recommended to run inside a Docker container:

```bash
# Run inside of OvDSGG/
docker build -t ovdsgg:latest .
```
```bash
# Run directly outside of OvDSGG/
docker run --rm -it \
    --gpus all \
    --shm-size 16g \
    -e PYTHONPATH=/app/models/ops \
    -v /path/to/OvDSGG:/app \
    -v /path/to/data:/app/data \
    -v /path/to/exps:/app/exps \
    ovdsgg:latest
```

Build MultiScaleDeformableAttention

```
cd models/ops
python setup.py build install --user
```
Verify the build:

`python -c "from models.ops.modules import MSDeformAttn; print('OK')"
`

## Training
Model is trained in three stages. Each stage initialises form previous stage's final checkpoint.
You can see checkpoints [here](https://drive.google.com/drive/folders/1vahEdkaiEz5m_tGhkO4bots758VtpQDl?usp=sharing).

To run a specific config (e.g. `r50_train_multi.sh`), run:

`GPUS_PER_NODE=2 ./tools/run_dist_launch.sh $1 r50 $2 configs/r50_train_multi.sh`

## Evaluation
Evlaution is single-GPU so as to not trigger NCCL timeouts.
Run above command, but with `GPUS_PER_NODE=1` and `CUDA_VISIBLE_DEVICES=0` or whichever GPU you want to show.

Main flag is `--eval` alongside others in `r50_eval_multi.sh`, where `--resume` is the checkpoint to evaluate from.
