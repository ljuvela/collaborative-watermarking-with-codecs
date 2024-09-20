# Collaborative Watermarking with Codecs

This repository contains code and supplementary material related to our recent paper **Audio Codec Augmentation for Robust Collaborative Watermarking of Speech Synthesis**
(Submitted to ICASSP 2025)

Paper pre-print will be made available once the supplementary material is complete.

Audio samples are  available at the demo page
http://ljuvela.github.io/collaborative-watermarking-with-codecs-demo

Model checkpoints are available be available at 
[https://huggingface.co/ljuvela/collaborative-watermarking-with-codecs](https://huggingface.co/ljuvela/checkpoints-for-collaborative-watermarking-with-codecs)

## Environment setup


Create a new conda environment with the provided environment file (using mamba package manager):
```bash
mamba env create -n collaborative-watermarking-with-codecs -f pytorch-env.yml
mamba activate collaborative-watermarking-with-codecs
```

### Pre-trained models for initialization

Pre-trained models are distributed with Git LFS and included as submodules. To download the pre-trained models, run the following command:
```bash
git submodule update --init --recursive
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

## Dataset 

LibriTTS-R dataset was used for all experiments. The dataset is available at:
http://www.openslr.org/141/

This repository includes some wav files from the dataset for demonstration purposes (as per CC BY 4.0 licence). For full replication, the full dataset should be downloaded separately.

### Rendering audio with pre-trained models

You can download all the models used in experiments by cloning the following repository:
```bash
git clone https://huggingface.co/ljuvela/checkpoints-for-collaborative-watermarking-with-codecs
```

Assuming the checkpoints are not downloaded in the current directory, you can render audio with the following command:
```bash
model_id=collab-dac # or any other model id
checkpoints_dir=./checkpoints-for-collaborative-watermarking-with-codecs/$model_id/
python src/collaborative_watermarking/render/render_hifigan.py \
    --config $checkpoints_dir/config.json \
    --input_file data/input_wavs_demo/filelist.txt \
    --input_wavs_dir data/input_wavs_demo \
    --output_wavs_dir output/$model_id \
    --checkpoint_path $checkpoints_dir \
    --wavefile_ext ""
```



### Training models


Example Slurm batch script

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
conda activate collaborative-watermarking-with-codecs

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




