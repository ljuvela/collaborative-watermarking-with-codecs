#!/bin/zsh
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=VCTK-download

source ~/.zshrc
module load mamba
# source /appl/scibuilder-mamba/aalto-rhel9/prod/software/mamba/2024-01/39cf5e1/etc/profile.d/mamba.sh 
conda activate CollaborativeWatermarking-2024

export DAREA_DATA_PATH='/scratch/elec/t412-speechsynth/DATA'

cd /scratch/elec/t412-speechsynth/ljuvela/CODE/collaborative-watermarking

python scripts/download_vctk.py --target_dir=$DAREA_DATA_PATH