#!/bin/bash
#SBATCH -J bml-pred
#SBATCH --ntasks=1
#SBATCH --time=0-01:30:00
#-----SBATCH --array=3,6
#-----SBATCH --array=0-11
#-----SBATCH --gres=gpu:1
#SBATCH -p dgx2q

echo "Loading modules"
#module use /cm/shared/ex3-modules/latest/modulefiles
#source /etc/profile.d/modules.sh
module load slurm/20.02.7
module load tensorflow2-py37-cuda10.2-gcc8/2.5.0

# Activate venv -- not required due to site_packages
#source bml-env/bin/activate

python --version

srun python predict.py csv/building-list-Bjerke.csv predictions/preds-Bjerke.csv

