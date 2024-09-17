import os
import argparse

filelist_dir = 'src/collaborative_watermarking/third_party/vits/filelists'
vits_filelist_train = os.path.join(filelist_dir, 'ljs_audio_text_train_filelist.txt.cleaned')
vits_filelist_val = os.path.join(filelist_dir, 'ljs_audio_text_val_filelist.txt.cleaned')
vits_filelist_test = os.path.join(filelist_dir, 'ljs_audio_text_test_filelist.txt.cleaned')

def clean_vits_filelist(infile, outfile, prefix='', remove_text=True):

    # make sure prefix ends with '/'
    if prefix and not prefix.endswith('/'):
        prefix += '/'

    with open(infile, 'r') as f:
        lines = f.readlines()
        with open(outfile, 'w') as g:
            for line in lines:
                line = line.strip()
                path, text = line.split('|')
                # replace 'DUMMY1/' with prefix
                path = path.replace('DUMMY1/', prefix)

                if prefix and not os.path.exists(path):
                    print(f'File not found: {path}, skipping')
                    continue

                if remove_text:
                    g.write(path + '\n')
                else:
                    g.write(path + '|' + text + '\n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Clean VITS LJSpeech filelist')
    parser.add_argument('--target_dir', type=str, default='experiments/filelists/vctk')
    parser.add_argument('--prefix', type=str, default='')
    args = parser.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)

    # Keep text and append prefix (local dirs)
    clean_vits_filelist(
        vits_filelist_train,
        os.path.join(args.target_dir, "ljspeech_filelist_train_text_local.txt"),
        prefix=os.path.join(args.prefix, "train"),
        remove_text=False,
    )
    clean_vits_filelist(
        vits_filelist_val,
        os.path.join(args.target_dir, "ljspeech_filelist_val_text_local.txt"),
        prefix=os.path.join(args.prefix, "val"),
        remove_text=False,
    )
    clean_vits_filelist(
        vits_filelist_test,
        os.path.join(args.target_dir, "ljspeech_filelist_test_text_local.txt"),
        prefix=os.path.join(args.prefix, "test"),
        remove_text=False,
    )

    print('VITS LJSpeech filelists cleaned and saved to {}'.format(args.target_dir))
