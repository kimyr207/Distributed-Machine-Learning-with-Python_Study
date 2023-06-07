#! /bin/bash
#SBATCH --job-name=multinode_torchrun
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%j.%x.log
#SBATCH --error=logs/%j.%x.log

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$(expr 20000 + $(echo -n $SLURM_JOBID | tail -c 4))

export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=32

export LAUNCHER="torchrun \
	--nnodes $SLURM_NNODES \
	--nproc_per_node 8 \
	--rdzv_id $UID \
	--rdzv_backend c10d \
	--rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
        "

export RUN_CMD="/t1data/users/alan.kim/project/python_dis/ch3/multnode_torchrun.py 6 2"

module purge
module load Miniconda3/22.11.1-1
source activate pytorch20

srun bash -c "$LAUNCHER $RUN_CMD"
