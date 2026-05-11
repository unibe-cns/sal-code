#!/usr/bin/env bash
# slurm_submit.sh — Step 2 of the SLURM sweep workflow.
#
# Prerequisites:
#   jobs.sh must exist (created by sweep_creator.py).
#   slurm.sh must exist and contain your #SBATCH directives.
#
# What this script does:
#   1. Snapshots jobs.sh to jobs_<timestamp>.sh so that late-running array
#      tasks read a stable file even if jobs.sh is regenerated in the meantime.
#   2. Submits a SLURM array job (one task per line in the snapshot).
#      Each task reads its command from the snapshot via $SLURM_ARRAY_TASK_ID
#      and executes it (see slurm.sh).
#
# Usage:
#   bash slurm_submit.sh

# snapshot a copy of jobs.sh
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SNAPSHOT="jobs_${TIMESTAMP}.sh"

cp jobs.sh "${SNAPSHOT}" 
echo "Copy a snapshot of jobs.sh to ${SNAPSHOT}"
TOTAL=$(wc -l < "${SNAPSHOT}")

sbatch --array=1-${TOTAL} --export="ALL,JOBS_FILE=${SNAPSHOT}" slurm.sh
echo "Submitted ${TOTAL} jobs."
