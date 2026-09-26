#!/bin/bash
# =============================================================================
# main_job_vjepa2_1.sh — IntPhys2 V-JEPA 2.1 (ViT-g/16, 384px) launcher for RCS
#
# Submitted via Open OnDemand Job Composer.
# This script:
#   1. Clones IntPhys2 from GitHub (or pulls if already cloned)
#   2. Creates / reuses a Python venv and installs requirements
#   3. Downloads the V-JEPA 2.1 checkpoint (~17 GB) into $TORCH_HOME if missing
#   4. Launches main_distributed.py, which uses submitit to submit the real
#      GPU evaluation job into SLURM: 1-6 nodes (every node with a >=16 GB
#      GPU), sized to whatever is free when it starts.
#
# JOB LIMIT: max 2 jobs per user (queued + running, Open OnDemand sessions
# count). This launcher + the eval job it submits = 2. Close any OOD
# Jupyter / VS Code / RStudio session first, or the eval sbatch is REFUSED.
# =============================================================================

#SBATCH --job-name=vjepa2_1_intphys2
#SBATCH --partition=gpu
#SBATCH --account=grp-asem.a.abdelaziz
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/home/mohamed.gad04/logs/vjepa2_1_launcher_%j.out
#SBATCH --error=/home/mohamed.gad04/logs/vjepa2_1_launcher_%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Cluster identity (see main_job.sh)
# ---------------------------------------------------------------------------
CLUSTER_NAME="rcs"
export SLURM_CLUSTER_NAME="$CLUSTER_NAME"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_URL="https://github.com/Gad-MA/IntPhys2.git"
PROJECT_ROOT="$HOME/IntPhys2"
EVAL_ROOT="$PROJECT_ROOT/prediction_evals"
VENV_DIR="$HOME/venvs/IntPhys2"
# !! Switch between the fpc64 and fpc48 configs here !!
CONFIG="$EVAL_ROOT/evals/intphys2/configs/vjepa2_1_vitg_384_fpc48.yaml"
SUBMITIT_LOGS="$EVAL_ROOT/logs/submitit"
RESULTS_DIR="$EVAL_ROOT/results/vjepa2_1_vitg_384"

# Eval job resources (per node). Must fit the smallest eligible node
# (gpu-01/02/04: 29000 MB schedulable, 12 CPUs; submitit asks 10 CPUs/node).
EVAL_MEM="24G"
# Minutes. Must cover the WORST case, i.e. min_nodes (=1) on the slowest
# 16 GB card: unmeasured estimate ~8 h for intphys2-debug. Check `sacct`
# Elapsed/NodeList after the first run, then lower this or raise min_nodes
# in the config (a shorter request starts sooner). gpu max: 4320.
EVAL_TIME_MIN=600
# GPUs below 16 GB: gpu-03 (GTX 1080 Ti 11 GB + RTX 3060 12 GB), gpu-08 (GTX 1080 Ti 11 GB)
EVAL_EXCLUDE="gpu-03,gpu-08"

export HF_HOME="/data/asem.a.abdelaziz/gp-2027a/huggingface_downloads"
export TORCH_HOME="/data/asem.a.abdelaziz/gp-2027a/torch_downloads"
CKPT_URL="https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitg_384.pt"
# Must match the path the wrapper resolves: torch.hub.get_dir()/checkpoints/<basename>
CKPT_PATH="$TORCH_HOME/hub/checkpoints/$(basename "$CKPT_URL")"

mkdir -p "$HOME/logs"

echo "=========================================="
echo " IntPhys2 V-JEPA 2.1 benchmark launcher"
echo " Cluster    : $CLUSTER_NAME"
echo " Project    : $PROJECT_ROOT"
echo " Venv       : $VENV_DIR"
echo " Config     : $CONFIG"
echo " Checkpoint : $CKPT_PATH"
echo " Logs       : $SUBMITIT_LOGS"
echo " Results    : $RESULTS_DIR"
echo " Started    : $(date)"
echo "=========================================="

# ---------------------------------------------------------------------------
# 1. Clone or pull the repository
# ---------------------------------------------------------------------------
echo ""
echo "[1/4] Setting up repository..."

if [ -d "$PROJECT_ROOT/.git" ]; then
    echo "  Repo already exists — pulling latest changes..."
    git -C "$PROJECT_ROOT" pull --ff-only
else
    echo "  Cloning $REPO_URL → $PROJECT_ROOT ..."
    git clone "$REPO_URL" "$PROJECT_ROOT"
fi

echo "  Repo is up to date. (commit: $(git -C "$PROJECT_ROOT" rev-parse --short HEAD))"

# ---------------------------------------------------------------------------
# 2. Create / reuse the virtual environment
# ---------------------------------------------------------------------------
echo ""
echo "[2/4] Setting up Python environment..."

if [ ! -d "$VENV_DIR" ]; then
    echo "  Creating venv at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
else
    echo "  Venv already exists — reusing."
fi

source "$VENV_DIR/bin/activate"
echo "  Python: $(python --version)  |  pip: $(pip --version | awk '{print $2}')"

echo "  Installing / updating requirements..."
pip install -r "$EVAL_ROOT/requirements.txt"
echo "  Requirements satisfied."

# ---------------------------------------------------------------------------
# 3. Download the checkpoint once (so the 6 GPU ranks don't all download it)
# ---------------------------------------------------------------------------
echo ""
echo "[3/4] Checking V-JEPA 2.1 checkpoint..."

mkdir -p "$(dirname "$CKPT_PATH")"
if [ -f "$CKPT_PATH" ]; then
    echo "  Found $CKPT_PATH — skipping download."
else
    echo "  Downloading $CKPT_URL (~17 GB, resumable)..."
    wget -c -q --show-progress -O "$CKPT_PATH.part" "$CKPT_URL"
    mv "$CKPT_PATH.part" "$CKPT_PATH"
    echo "  Download complete."
fi

# ---------------------------------------------------------------------------
# 4. Launch the distributed evaluation
# ---------------------------------------------------------------------------
echo ""
echo "[4/4] Submitting evaluation job via submitit..."
echo "  submitit will request 1-6 nodes x 1 GPU (whatever is free at start), $EVAL_MEM RAM/node, $EVAL_TIME_MIN min."
echo ""

mkdir -p "$SUBMITIT_LOGS" "$RESULTS_DIR"
cd "$EVAL_ROOT"

python -m evals.main_distributed \
    --fname   "$CONFIG"              \
    --folder  "$SUBMITIT_LOGS"       \
    --partition gpu                  \
    --account grp-asem.a.abdelaziz  \
    --qos     normal                 \
    --time    "$EVAL_TIME_MIN"       \
    --mem     "$EVAL_MEM"            \
    --exclude "$EVAL_EXCLUDE"

echo ""
echo "=========================================="
echo " Launcher finished at $(date)."
echo " Real eval job IDs are in: $SUBMITIT_LOGS"
echo "=========================================="
