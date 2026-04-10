#!/usr/bin/env bash
# Run all SSN training jobs for this experiment folder.
#
# Usage:
#   bash run.sh [NUM_SEEDS, default: 5]
#
# Steps:
#   1. Regenerate per-sweep parameter files via change_params.py
#   2. Launch every (sweep_id, seed_id) combination as a background process
#   3. Wait for all processes to finish
#
# SLURM compatibility: call this script from a SLURM job script.
# Each background process runs on one of the job's allocated cores.
#
# Results are written to results/ssn/<exp_type>/ relative to the repo root.
set -euo pipefail

# Number of random seeds per sweep (override via first positional argument)
NUM_SEEDS=${1:-5}

# Resolve paths relative to this script so it can be called from anywhere
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
TRAIN="$SCRIPT_DIR/../train_bm_hpo.py"

# --- Step 1: generate per-sweep parameter files ---
echo "Generating parameter files..."
cd "$SCRIPT_DIR"
python change_params.py

# --- Step 2: launch all runs in parallel ---
# discover the generated sweep files
mapfile -t SWEEP_FILES < <(ls "$SCRIPT_DIR"/exp.[0-9]*.yaml 2>/dev/null | sort)
NUM_SWEEPS=${#SWEEP_FILES[@]}
echo "Found $NUM_SWEEPS sweep(s) × $NUM_SEEDS seed(s) = $((NUM_SWEEPS * NUM_SEEDS)) total runs"

# iterate over the sweep files and seeds and launch them
for sweep_file in "${SWEEP_FILES[@]}"; do
    stem=$(basename "$sweep_file" .yaml)   # e.g. exp.0002
    sweep_id=$((10#${stem#exp.}))           # strip "exp." prefix, force base-10
    for seed_id in $(seq 0 $((NUM_SEEDS - 1))); do
        python "$TRAIN" "$sweep_file" -i "$sweep_id" -s "$seed_id" -o "$SCRIPT_DIR/../../../results/ssn/"&
    done
done

# --- Step 3: wait for all background jobs ---
wait
echo "All $((NUM_SWEEPS * NUM_SEEDS)) runs finished."
