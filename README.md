# Collaborative Watermarking

Submitted to ICASSP 2025.


## Environment setup


Create a new conda environment with the provided environment file (using mamba package manager):
```bash
mamba env create -n collaborative-watermarking-with-codecs -f pytorch-env.yml
mamba activate collaborative-watermarking-with-codecs
```

### Installing DAREA
Install differentiable augmentation and robustness evaluation package.

Follow the insallation instructions at:
https://github.com/ljuvela/DAREA

This should be installed in the same environment as the collaborative watermarking package.


### Installing DAC
```
git submodule update --init --recursive
cd src/collaborative_watermarking/third_party/dac
pip install -e ".[dev]"
```

Note that the pesq package needs to compile C/C++ extensions and requires `gcc` or similar compiler on the system.




Set environment variables:

```bash
export DAREA_DATA_PATH=/path/to/data
```

Install the package in editable mode:
```bash
pip install -e .
```





Run unit tests
```bash
pytest -s tests
```






### Example Slurm batch script


```bash
#!/bin/zsh
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=5
#SBATCH --job-name=Job-1

# Set job ID
ID=01

# Activate environment
source ~/.zshrc
module load mamba
conda activate CollaborativeWatermarking2024

# Set DAREA data path
export DAREA_DATA_PATH='<path-to-data>'

# Set current working directory
cd <path-to-source-code>/collaborative-watermarking

# Set filelists and configs
train_file=<path-to-source-code>/collaborative-watermarking/src/collaborative_watermarking/filelists/vctk/vctk_filelist_mic2_train.txt
valid_file=<path-to-source-code>/collaborative-watermarking/src/collaborative_watermarking/filelists/vctk/vctk_filelist_mic2_val.txt
config_file=<path-to-source-code>/collaborative-watermarking/experiments/$ID/config_v1.json
vctk_dir=<path-to-data>/torchaudio/VCTK-Corpus-0.92/wav48_silence_trimmed
pretrained_hifigan_path=<path-to-data>/pretrained/hifigan/UNIVERSAL_V1

# total num epochs counts when continuing training
epochs=100

python src/collaborative_watermarking/train/train_hifigan.py \
    --config $config_file \
    --input_training_file $train_file \
    --input_validation_file $valid_file \
    --input_wavs_dir $vctk_dir \
    --pretrained_hifigan_path $pretrained_hifigan_path \
    --training_epochs $epochs \
    --wavefile_ext "" \
    --validation_interval 2000 \
    --summary_interval 200 \
    --log_training_eer True \
    --use_augmentation False \
    --checkpoint_path "ckpt/cp_hifigan_$ID"
```




