#!/bin/bash
# =============================================================================
# main_job_vjepa2_1.sh — IntPhys2 V-JEPA 2.1 (ViT-g/16, 384px) launcher for RCS
#
# Submit from the repo root (or the Open OnDemand Job Composer):
#   sbatch main_job_vjepa2_1.sh                                     # Debug
#   sbatch --export=ALL,DATASET=intphys2-main main_job_vjepa2_1.sh  # Main
# Optional overrides: CONFIG, MIN_NODES, EVAL_TIME_MIN.
# This script:
#   1. Clones IntPhys2 from GitHub (or pulls if already cloned)
#   2. Creates / reuses a Python venv and installs requirements
#   3. Downloads the V-JEPA 2.1 checkpoint (~17 GB) into $TORCH_HOME if missing
#   4. Launches main_distributed.py, which uses submitit to submit the real
#      GPU evaluation job into SLURM: MIN_NODES-6 nodes (every node with a
#      >=16 GB GPU), sized to whatever is free when it starts.
#
# JOB LIMIT: max 2 jobs per user (queued + running, Open OnDemand sessions
# count). This launcher + the eval job it submits = 2. Close any OOD
# Jupyter / VS Code / RStudio session first, or the eval sbatch is REFUSED.
# =============================================================================

#SBATCH --job-name=vjepa2_1_intphys2
#SBATCH --partition=gpu
#SBATCH --account=gp-2027-asem.a.abdelaziz-a
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=3-00:00:00
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
CONFIG="${CONFIG:-$EVAL_ROOT/evals/intphys2/configs/vjepa2_1_vitg_384_fpc48.yaml}"
DATASET="${DATASET:-intphys2-debug}"   # intphys2-debug | intphys2-main | intphys2-heldout
SUBMITIT_LOGS="$EVAL_ROOT/logs/submitit"
RESULTS_DIR="$EVAL_ROOT/results/vjepa2_1_vitg_384"

# Eval job resources (per node). Must fit the smallest eligible node
# (gpu-01/02/04: 29000 MB schedulable, 12 CPUs; submitit asks 10 CPUs/node).
EVAL_MEM="24G"
# Fewest nodes to start with, sized so the WORST case (MIN_NODES nodes, all
# on the slowest 16 GB card) fits well inside the time limit. Measured by
# check_vjepa2_1_rcs.sh (job 1613, RTX 5060 Ti): ~6.6 min/video, so
#   debug   (60 videos):   ~6.6 GPU-h -> 1 node  ~6.6 h  (6 nodes ~1.2 h)
#   main    (1012 videos): ~115 GPU-h -> 3 nodes ~39 h   (6 nodes ~20 h; 1 node > 72 h)
#   heldout (344 videos):  ~38 GPU-h  -> 1 node  ~38 h
case "$DATASET" in
    intphys2-debug)   DEFAULT_MIN_NODES=1 ;;
    intphys2-main)    DEFAULT_MIN_NODES=3 ;;
    intphys2-heldout) DEFAULT_MIN_NODES=1 ;;
    *) echo "Unknown DATASET=$DATASET" >&2; exit 1 ;;
esac
MIN_NODES="${MIN_NODES:-$DEFAULT_MIN_NODES}"
# Minutes. Always the gpu partition max (3 days) so a slow run is never cut
# off; the job still ends as soon as the eval finishes.
EVAL_TIME_MIN="${EVAL_TIME_MIN:-4320}"
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
echo " Dataset    : $DATASET  (min_nodes $MIN_NODES, $EVAL_TIME_MIN min)"
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

# Per-run copy of the config with the requested dataset and min_nodes
RUN_CONFIG_DIR="$EVAL_ROOT/logs/run_configs"
RUN_CONFIG="$RUN_CONFIG_DIR/$(basename "${CONFIG%.yaml}")_${DATASET}_${SLURM_JOB_ID:-local}.yaml"
mkdir -p "$RUN_CONFIG_DIR"
python - "$CONFIG" "$RUN_CONFIG" "$DATASET" "$MIN_NODES" <<'EOF'
import sys, yaml
src, dst, dataset, min_nodes = sys.argv[1:]
cfg = yaml.safe_load(open(src))
cfg["experiment"]["data"]["dataset"] = dataset
cfg["min_nodes"] = int(min_nodes)
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
EOF
NODES=$(python -c "import yaml; print(yaml.safe_load(open('$RUN_CONFIG'))['nodes'])")
echo "  Run config: $RUN_CONFIG"
echo "  submitit will request $MIN_NODES-$NODES nodes x 1 GPU (whatever is free at start), $EVAL_MEM RAM/node, $EVAL_TIME_MIN min."
echo ""

mkdir -p "$SUBMITIT_LOGS" "$RESULTS_DIR"
cd "$EVAL_ROOT"

python -m evals.main_distributed \
    --fname   "$RUN_CONFIG"          \
    --folder  "$SUBMITIT_LOGS"       \
    --partition gpu                  \
    --account gp-2027-asem.a.abdelaziz-a  \
    --qos     normal                 \
    --time    "$EVAL_TIME_MIN"       \
    --mem     "$EVAL_MEM"            \
    --exclude "$EVAL_EXCLUDE"

echo ""
echo "=========================================="
echo " Launcher finished at $(date)."
echo " Real eval job IDs are in: $SUBMITIT_LOGS"
echo "=========================================="
