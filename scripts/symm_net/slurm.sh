#!/bin/bash
# --- HPC cluster specific SLURM settings ------------------------
# add your settings here.
#SBATCH --job-name="symmnet sweep"
#SBATCH --time=<xx:xx:xx>
#SBATCH ...
# ----------------------------------------------------------------

# --- HPC cluster specific setup:---------------------------------
# e.g. load modules
# and activate python environment
# ----------------------------------------------------------------

echo "Start job with id ${SLURM_ARRAY_TASK_ID}."

# read the snapshotted jobs_<timestamp>.sh file
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${JOBS_FILE}")
echo "Run command ${CMD}"
# and execute it.
eval "$CMD"
echo "Done."
