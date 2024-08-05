import os
from glob import glob

basedir = '/scratch/elec/t412-speechsynth/DATA/LibriTTS-R/LibriTTS_R'

filelist_dir = 'src/collaborative_watermarking/filelists/libritts'

train_dir = os.path.join(basedir, 'train-clean-100')
valid_dir = os.path.join(basedir, 'dev-clean')
test_dir = os.path.join(basedir, 'test-clean')

# find all wav files recursively
train_files = []
for ext in ['**/*.flac', '**/*.wav']:
    train_files.extend(glob(os.path.join(train_dir, ext), recursive=True))

# remove the prefix
train_files = [f.replace(basedir + '/', '') for f in train_files]

# write to file
with open(os.path.join(filelist_dir, 'libritts_filelist_train.txt'), 'w') as f:
    for file in train_files:
        f.write(file + '\n')

# find all wav files recursively
valid_files = []
for ext in ['**/*.flac', '**/*.wav']:
    valid_files.extend(glob(os.path.join(valid_dir, ext), recursive=True))

# remove the prefix
valid_files = [f.replace(basedir + '/', '') for f in valid_files]

# write to file
with open(os.path.join(filelist_dir, 'libritts_filelist_val.txt'), 'w') as f:
    for file in valid_files:
        f.write(file + '\n')

# find all wav files recursively
test_files = []
for ext in ['**/*.flac', '**/*.wav']:
    test_files.extend(glob(os.path.join(test_dir, ext), recursive=True))

# remove the prefix
test_files = [f.replace(basedir + '/', '') for f in test_files]

# write to file
with open(os.path.join(filelist_dir, 'libritts_filelist_test.txt'), 'w') as f:
    for file in test_files:
        f.write(file + '\n')


