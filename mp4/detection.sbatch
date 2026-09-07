#!/bin/bash

### Do not change the following 8 lines.
#SBATCH --job-name=detection
#SBATCH --account=bfdu-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=48g
### End Do not change block

#SBATCH --mail-user=NETID@illinois.edu  # replace with your email
#SBATCH --mail-type=NONE                # Type of email notifications to send

#SBATCH --output=ece494mp4-%x-%j.out  # Default name of batch job output file if not set on command line
#SBATCH --error=ece494mp4-%x-%j.err   # Default name of batch job error file if not set on command line

# This creates output directory for every run. Change EXP_NAME and DIR_NAME
# appropriately while performing each run and ablation experiment

# Default values for DIR_NAME and EXP_NAME. For each different experiment you
# run, you should use a different EXP_NAME. Change it in the command line flag.
OUTPUT_DIR=${OUTPUT_DIR:="runs/run2"}
mkdir -p ${OUTPUT_DIR}
LOG_FILE="${OUTPUT_DIR}/log.txt"

# Activate your venv
source /work/nvme/bfdu/cs444/fa25/mp4/venv/bin/activate

# print some diagnostics
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true
export PYTHONUNBUFFERED=1

# Run your job 
python demo.py --seed 1 --lr 1e-2 --batch_size 1 --output_dir ${OUTPUT_DIR} --coco_dir /work/nvme/bfdu/cs444/fa25/mp4/data/coco-animal 2>&1 | tee ${LOG_FILE}
