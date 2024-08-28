#!/bin/bash
# TODO update settings based on your needs and computational resources provider (here 1 nodes * 1 gpus = 1 gpu):
#SBATCH --job-name=asdf_job
#SBATCH --account=project_2006687
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=1500
#SBATCH --time=0:15:00
#SBATCH --gres=gpu:v100:1,nvme:85

module purge
module load pytorch

# Copies dataset to SSD drive
tar xf /scratch/project_2006687/datasets/tars/LA.tar -C $LOCAL_SCRATCH
#tar xf /scratch/project_2006687/datasets/tars/asvspoof2021.tar -C $LOCAL_SCRATCH
#tar xf /scratch/project_2006687/datasets/tars/MLAADv3_16khz.tar -C $LOCAL_SCRATCH
#tar xf /scratch/project_2006687/datasets/tars/m-ailabs-mlaad-sources.tar -C $LOCAL_SCRATCH
#tar xf /scratch/project_2006687/datasets/tars/WaveFake_16khz.tar -C $LOCAL_SCRATCH
#tar xf /scratch/project_2006687/datasets/tars/in_the_wild.tar -C $LOCAL_SCRATCH
#tar xf /scratch/project_2006687/datasets/tars/for-norm.tar -C $LOCAL_SCRATCH


srun python -u ../asdf/recipes/wav2vec_aasist/run.py ${*%${!#}} > ${@:$#}

# The last argument is the output filename, the ones before that are run configs.

