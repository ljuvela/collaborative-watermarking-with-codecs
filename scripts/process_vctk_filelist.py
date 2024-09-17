import os
import argparse

filelist_dir = 'src/collaborative_watermarking/third_party/vits/filelists'
vits_filelist_train = os.path.join(filelist_dir, 'vctk_audio_sid_text_train_filelist.txt.cleaned')
vits_filelist_val = os.path.join(filelist_dir, 'vctk_audio_sid_text_val_filelist.txt.cleaned')
vits_filelist_test = os.path.join(filelist_dir, 'vctk_audio_sid_text_test_filelist.txt.cleaned')

def clean_vits_filelist(infile, outfile, prefix='', remove_text=True):

    # make sure prefix ends with '/'
    if prefix and not prefix.endswith('/'):
        prefix += '/'
    with open(infile, 'r') as f:
        lines = f.readlines()
        with open(outfile, 'w') as g:
            for line in lines:
                line = line.strip()
                path, sid, text = line.split('|')
                # replace 'DUMMY2/' with prefix
                path = path.replace('DUMMY2/', prefix)
                # p255_157_mic2.flac
                path = path.replace('.wav', '_mic2.flac')

                if prefix and not os.path.exists(path):
                    print(f'File not found: {path}, skipping')
                    continue
                if remove_text:
                    g.write(path + '\n')
                else:
                    g.write(path + '|' + sid + '|' + text + '\n')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Clean VITS VCTK filelist')
    parser.add_argument('--target_dir', type=str, default='experiments/filelists/vctk')
    parser.add_argument('--prefix', type=str, default='')
    args = parser.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)

    # Keep text and append prefix
    clean_vits_filelist(vits_filelist_train, os.path.join(args.target_dir, 'vctk_filelist_train_text.txt'), prefix=args.prefix, remove_text=False)
    clean_vits_filelist(vits_filelist_val, os.path.join(args.target_dir, 'vctk_filelist_val_text.txt'), prefix=args.prefix, remove_text=False)
    clean_vits_filelist(vits_filelist_test, os.path.join(args.target_dir, 'vctk_filelist_test_text.txt'), prefix=args.prefix, remove_text=False)

    # Split text and remove prefix
    clean_vits_filelist(vits_filelist_train, os.path.join(args.target_dir, 'vctk_filelist_train.txt'))
    clean_vits_filelist(vits_filelist_val, os.path.join(args.target_dir, 'vctk_filelist_val.txt'))
    clean_vits_filelist(vits_filelist_test, os.path.join(args.target_dir, 'vctk_filelist_test.txt'))
    

    # Keep text and append prefix (local dirs)
    clean_vits_filelist(
        vits_filelist_train,
        os.path.join(args.target_dir, "vctk_filelist_train_text_local.txt"),
        prefix="/tmp/ljuvela/data/vctk/train",
        remove_text=False,
    )
    clean_vits_filelist(
        vits_filelist_val,
        os.path.join(args.target_dir, "vctk_filelist_val_text_local.txt"),
        prefix="/tmp/ljuvela/data/vctk/val",
        remove_text=False,
    )
    clean_vits_filelist(
        vits_filelist_test,
        os.path.join(args.target_dir, "vctk_filelist_test_text_local.txt"),
        prefix="/tmp/ljuvela/data/vctk/test",
        remove_text=False,
    )
    
    
    
    print('VITS VCTK filelists cleaned and saved to {}'.format(args.target_dir))