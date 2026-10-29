#!/bin/bash
# =============================================================================
# check_vjepa2_1_rcs.sh — VRAM + speed check for V-JEPA 2.1 ViT-g/16 @ 384px
# on a 16 GB RCS card, before launching main_job_vjepa2_1.sh.
#
# Builds the model exactly as the IntPhys2 eval does (same wrapper, config,
# bf16 autocast, no_grad) and times forward passes on random clips of the
# config's shape (48 frames, 384px, chunk_size windows per pass). Prints:
#   - peak VRAM per pass type and the headroom left on the card,
#   - seconds per pass, and an estimate of GPU-hours for Debug / Main.
#
# Runs on gpu-02 (RTX 5060 Ti, 16,311 MiB: the smallest 16 GB card). If it
# fits there it fits on gpu-01 / gpu-04 too.
# Reuses the launcher's repo, venv and checkpoint paths; downloads the
# checkpoint (~17 GB, resumable) only if it is not there yet.
#
# JOB LIMIT: counts as 1 of your 2 jobs. Don't run main_job_vjepa2_1.sh
# (launcher + eval = 2 jobs) at the same time.
# =============================================================================

#SBATCH --job-name=vjepa2_1_vramcheck
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx5060ti:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/mohamed.gad04/logs/vjepa2_1_vramcheck_%j.out
#SBATCH --error=/home/mohamed.gad04/logs/vjepa2_1_vramcheck_%j.err

set -euo pipefail

# Same paths as main_job_vjepa2_1.sh
REPO_URL="https://github.com/Gad-MA/IntPhys2.git"
PROJECT_ROOT="$HOME/IntPhys2"
EVAL_ROOT="$PROJECT_ROOT/prediction_evals"
VENV_DIR="$HOME/venvs/IntPhys2"
CONFIG="$EVAL_ROOT/evals/intphys2/configs/vjepa2_1_vitg_384_fpc48.yaml"
export TORCH_HOME="/data/asem.a.abdelaziz/gp-2027a/torch_downloads"
CKPT_URL="https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitg_384.pt"
CKPT_PATH="$TORCH_HOME/hub/checkpoints/$(basename "$CKPT_URL")"

mkdir -p "$HOME/logs"
echo "Node: $(hostname)  Job: $SLURM_JOB_ID  Started: $(date)"

# -- repo (needs app/vjepa_2_1 from commit d052b8a or later)
if [ -d "$PROJECT_ROOT/.git" ]; then
    git -C "$PROJECT_ROOT" pull --ff-only
else
    git clone "$REPO_URL" "$PROJECT_ROOT"
fi
echo "Commit: $(git -C "$PROJECT_ROOT" rev-parse --short HEAD)"

# -- venv (created by main_job.sh / main_job_vjepa2_1.sh)
if [ ! -x "$VENV_DIR/bin/python" ]; then
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install -r "$EVAL_ROOT/requirements.txt"
else
    source "$VENV_DIR/bin/activate"
fi

# -- checkpoint, at the path the wrapper resolves
mkdir -p "$(dirname "$CKPT_PATH")"
if [ ! -f "$CKPT_PATH" ]; then
    echo "Downloading $CKPT_URL (~17 GB, resumable)..."
    wget -c -q -O "$CKPT_PATH.part" "$CKPT_URL"
    mv "$CKPT_PATH.part" "$CKPT_PATH"
fi

cd "$EVAL_ROOT"
python - "$CONFIG" <<'EOF'
import sys
import time

import numpy as np
import torch
import yaml

from app.vjepa_2_1.modelcustom.default_wrapper import init_module

cfg = yaml.safe_load(open(sys.argv[1]))
d, mk = cfg["experiment"]["data"], cfg["model_kwargs"]
T, R, B = d["frames_per_clip"], d["resolution"], d["chunk_size"]
contexts = d["context_lengths"]

p = torch.cuda.get_device_properties(0)
total = p.total_memory / 2**30
print(f"GPU: {p.name}, {total:.2f} GiB | torch {torch.__version__} (CUDA {torch.version.cuda})")

t = time.time()
model = init_module(frames_per_clip=T, nb_context_frames=1, checkpoint=mk["checkpoint"],
                    model_kwargs=mk["pretrain_kwargs"], wrapper_kwargs=mk["wrapper_kwargs"])
model = model.cuda().eval()
for q in model.parameters():
    q.requires_grad = False
print(f"Model loaded in {time.time() - t:.0f}s, weights on GPU: {torch.cuda.memory_allocated() / 2**30:.2f} GiB")


@torch.no_grad()
def run(ctxt, b, reps=2):
    """Seconds per forward pass of b windows, peak allocated / reserved GiB."""
    model.nb_context_frames, model.frames_per_clip, model.grid_depth = ctxt, T, T // 2
    x = torch.randn(b, 3, T, R, R, device="cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x)
        torch.cuda.synchronize(); times.append(time.time() - t0)
    return min(times), torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30


worst = 0.0
print(f"\nSliding windows (chunk_size={B}):")
t_chunk = {}
for c in contexts:
    try:
        s, a, r = run(c, B)
    except torch.cuda.OutOfMemoryError:
        print(f"  ctxt {c:2d}: OUT OF MEMORY with chunk_size={B}")
        torch.cuda.empty_cache()
        s, a, r = run(c, 1)
        print(f"  ctxt {c:2d}: chunk_size=1 instead")
    t_chunk[c] = s
    worst = max(worst, r)
    print(f"  ctxt {c:2d}: {s:6.2f} s/pass  peak alloc {a:5.2f} GiB  reserved {r:5.2f} GiB")

print("\nEarly-context passes (max_context_mode, 1 window):")
probe = [2, max(contexts) // 4 * 2, max(contexts) - 2]  # even: multiples of tubelet_size
t_one = {}
for c in probe:
    s, a, r = run(c, 1)
    t_one[c] = s
    worst = max(worst, r)
    print(f"  ctxt {c:2d}: {s:6.2f} s/pass  peak alloc {a:5.2f} GiB  reserved {r:5.2f} GiB")

print(f"\nPeak reserved {worst:.2f} GiB of {total:.2f} GiB -> headroom {total - worst:.2f} GiB "
      f"({'OK' if total - worst > 0.5 else 'TOO TIGHT for 16 GB cards'})")

# Per-video cost: IntPhys2 videos are 636 frames @ 60 fps -> 64 frames at
# frame_step 10 -> (64 - 48) / stride + 1 = 9 windows per context length.
n_frames = int(np.ceil(636 / d["frame_steps"]))
n_win = (n_frames - T) // d["stride_sliding_window"] + 1
sec = sum(np.ceil(n_win / B) * t_chunk[c] for c in contexts)
sec += sum(np.interp(k, probe, [t_one[c] for c in probe])
           for c in contexts for k in range(2, c, 2))
print(f"\nEstimate on this card: {sec / 60:.1f} min/video "
      f"({n_win} windows x {len(contexts)} contexts + early-context passes)")
for name, n in (("Debug", 60), ("Main", 1012)):
    h = n * sec / 3600
    print(f"  {name:5s} ({n:4d} videos): {h:6.1f} GPU-h -> {h / 6:5.1f} h wall on 6 nodes, {h / 3:5.1f} h on 3")
EOF

echo ""
echo "Finished: $(date)"
