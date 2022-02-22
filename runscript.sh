#!/bin/sh
#SBATCH -J bml-cv
#SBATCH --ntasks=1
#SBATCH --time=0-04:00:00
#-----SBATCH --array=3,6
#SBATCH --array=0-11
#SBATCH --gres=gpu:1
#SBATCH -p dgx2q

echo "Loading modules"
module use /cm/shared/ex3-modules/latest/modulefiles
module load slurm/20.02.7
module load tensorflow2-py37-cuda10.2-gcc8/2.5.0

# Activate venv -- not required due to site_packages
#source bml-env/bin/activate

python --version

srun python cv_model_selection.py $SLURM_ARRAY_TASK_ID

