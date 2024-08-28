#!/bin/bash
# TODO update settings based on your needs and computational resources provider (here 2 nodes * 4 gpus = 8 gpus):
#SBATCH --job-name=asdf_job
#SBATCH --account=project_2006687
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=5000
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:v100:4,nvme:35

export RDZV_HOST=$(hostname)
export RDZV_PORT=29400

module purge
module load pytorch

# Copies dataset to SSD drive
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 tar xf /scratch/project_2006687/datasets/tars/LA.tar -C $LOCAL_SCRATCH
#srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 tar xf /scratch/project_2006687/datasets/tars/asvspoof2021.tar -C $LOCAL_SCRATCH
#srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 tar xf /scratch/project_2006687/datasets/tars/MLAADv3_16khz.tar -C $LOCAL_SCRATCH
#srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 tar xf /scratch/project_2006687/datasets/tars/m-ailabs-mlaad-sources.tar -C $LOCAL_SCRATCH
#srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 tar xf /scratch/project_2006687/datasets/tars/WaveFake_16khz.tar -C $LOCAL_SCRATCH
#srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 tar xf /scratch/project_2006687/datasets/tars/in_the_wild.tar -C $LOCAL_SCRATCH
#srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 tar xf /scratch/project_2006687/datasets/tars/for-norm.tar -C $LOCAL_SCRATCH

# TODO update nproc_per_node to match -gres=gpu:v100:4
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
    ../asdf/recipes/wav2vec_aasist/run.py ${*%${!#}} > ${@:$#}

# The last argument is the output filename, the ones before that are run configs.