#!/bin/bash
#SBATCH --partition=intelsr_short
#SBATCH --time=0-08:00:00
#SBATCH --ntasks=1
#SBATCH --account=ag_hiskp_funcke
##SBATCH --export=NONE

echo -e "Start $(date +"%F %T") | $SLURM_JOB_ID $SLURM_JOB_NAME | $(hostname) | $(pwd) \n" 

source ~/tf-cpu-env/bin/activate

python neuralFit.py --config params.json

echo -e "End $(date +"%F %T") | $SLURM_JOB_ID $SLURM_JOB_NAME | $(hostname) | $(pwd) \n" 
