

# read vits vctk filelist

import os
import argparse


vits_filelist_train = 'src/third_party/vits/filelists/vctk_audio_sid_text_train_filelist.txt'
vits_filelist_val = 'src/third_party/vits/filelists/vctk_audio_sid_text_val_filelist.txt'
vits_filelist_test = 'src/third_party/vits/filelists/vctk_audio_sid_text_test_filelist.txt'

def clean_vits_vctk_filelist(infile, outfile):
    with open(infile, 'r') as f:
        lines = f.readlines()
        with open(outfile, 'w') as g:
            for line in lines:
                line = line.strip()
                path = line.split('|')[0]
                # remove DUMMY2
                path = path.replace('DUMMY2/', '')
                # p255_157_mic2.flac
                path = path.replace('.wav', '')
                g.write(path + '\n')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Clean VITS VCTK filelist')
    parser.add_argument('--target_dir', type=str, default='experiments/filelists/vctk')
    args = parser.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)

    clean_vits_vctk_filelist(vits_filelist_train, os.path.join(args.target_dir, 'vctk_filelist_train.txt'))
    clean_vits_vctk_filelist(vits_filelist_val, os.path.join(args.target_dir, 'vctk_filelist_val.txt'))
    clean_vits_vctk_filelist(vits_filelist_test, os.path.join(args.target_dir, 'vctk_filelist_test.txt'))
    print('VITS VCTK filelists cleaned and saved to {}'.format(args.target_dir))