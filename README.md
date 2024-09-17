# Collaborative Watermarking


## Environment setup

```bash
mamba env create -n CollaborativeWatermarking2024 -f pytorch-env.yml
```

Install differentiable augmentation and robustness evaluation package
https://github.com/ljuvela/DAREA

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
pytest tests
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


### Installing DAC

```
git submodule update --init --recursive
cd src/collaborative_watermarking/third_party/dac
pip install -e ".[dev]"
```

Note that the pesq package needs to compile extensions and requires gcc or similar compiler on the system

Download pretrained models

```bash
python3 -m dac download --model_type 44khz # downloads the 44kHz variant
```

### VITS setup

Compile monotonic align
```bash
cd src/collaborative_watermarking/third_party/vits/monotonic_align
python setup.py build_ext --inplace
mkdir monotonic_align
cp build/lib.linux-x86_64-cpython-310/vits/monotonic_align monotonic_align

```


```bash
mkdir -p pre_trained/vits
gdown -O pre_trained/vits/ 'https://drive.google.com/uc?id=1q86w74Ygw2hNzYP9cWkeClGT5X25PvBT'
gdown -O pre_trained/vits/ 'https://drive.google.com/uc?id=11aHOlhnxzjpdWDpsz1vFDCzbeEfoIxru'
```

LJSpeech data preparation

Create filelists with absolute paths
```bash
python scripts/process_ljspeech_filelist.py --target_dir experiments/vits/filelists --prefix $DATA/LJSpeech-1.1/wavs/

python scripts/process_vctk_filelist.py --target_dir experiments/vits/filelists --prefix $DATA/torchaudio/VCTK-Corpus-0.92/wav48_silence_trimmed/

```



### Adding models to huggingface


Initial git lsf setup
```bash
git-lfs track "*.pth"
git add .gitattributes
```