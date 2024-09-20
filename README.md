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


Example Slurm batch script for training a model with DAC. This is a normal shell script, but the `#SBATCH` lines are used to specify the job parameters for the Slurm scheduler.

```bash
#!/bin/zsh
##SBATCH --time=48:00:00
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=5
#SBATCH --job-name=collab-dac

# Set job ID
ID=collab-dac

# Activate environment
source ~/.zshrc
module load mamba
conda activate CollaborativeWatermarking2024

# Set paths to match your system
DATA=<path/to/data/dir>
WORKDIR=<path/to/collaborative-watermarking-with-codecs>
# Set path to config file (for example DAC)
config_file=checkpoints-for-collaborative-watermarking-with-codecs/collab-dac/config.json
# Set path to LibriTTS-R dataset
libritts_dir=</path/to/LibriTTS_R/>

# Set current working directory
cd $WORKDIR

# Set filelists and configs (provided in the repository for LibriTTS-R)
train_file=$WORKDIR/src/collaborative_watermarking/filelists/libritts/libritts_filelist_train.txt 
valid_file=$WORKDIR/src/collaborative_watermarking/filelists/libritts/libritts_filelist_val.txt 

# pretrained baseline models (included in git lfs submodules)
pretrained_hifigan_path=pretrained/hifi-gan-baseline-model/UNIVERSAL_V1
pretrained_aasist_path=pretrained/aasist/AASIST.pth

# total num epochs counts for fine-tuning
epochs=20

python src/collaborative_watermarking/train/train_hifigan.py \
    --config $config_file \
    --input_training_file $train_file \
    --input_validation_file $valid_file \
    --input_wavs_dir $libritts_dir \
    --pretrained_hifigan_path $pretrained_hifigan_path \
    --pretrained_watermark_path $pretrained_aasist_path \
    --training_epochs $epochs \
    --wavefile_ext "" \
    --validation_interval 2000 \
    --summary_interval 200 \
    --checkpoint_path "ckpt/hifi_gan/$ID"

```




