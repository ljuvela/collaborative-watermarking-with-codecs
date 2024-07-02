import os

import argparse
import darea

from tqdm import tqdm

from importlib.resources import files

import darea.utils

filelists = {
    'train': files('collaborative_watermarking.filelists.vctk').joinpath('vctk_filelist_mic2_train.txt'),
    'val': files('collaborative_watermarking.filelists.vctk').joinpath('vctk_filelist_mic2_val.txt'),
    'test': files('collaborative_watermarking.filelists.vctk').joinpath('vctk_filelist_mic2_test.txt'),
}

def main(args):

    if args.input_dir is None:
         
        darea_data_path = darea.utils.get_data_path()
        args.input_dir = os.path.join(darea_data_path, "torchaudio", "VCTK-Corpus-0.92", "wav48_silence_trimmed")

    if args.output_dir is None:

        darea_data_path = darea.utils.get_data_path()
        output_dir = os.path.join(darea_data_path, "vctk")
    else:
        output_dir = args.output_dir


    for partition, filelist in filelists.items():

        print(f"Partition: {partition}")
        print(f"Input directory: {args.input_dir}")
        print(f"Output directory: {output_dir}")

        with open(filelist, "r") as f:
            lines = f.readlines()

        for line in tqdm(lines):

            line = line.strip()
            src = os.path.join(args.input_dir, line)
            dst = os.path.join(output_dir, partition, line)

            if not os.path.exists(src):
                raise FileNotFoundError(f"File does not exist: {src}")

            os.makedirs(os.path.dirname(dst), exist_ok=True)

            # copy file
            os.system(f"cp {src} {dst}")

            # # create soft link if it does not exist
            # if not os.path.exists(dst):
            #     os.symlink(src, dst)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create directories for VCTK partitions with symlinks to the original files.")
    parser.add_argument("--input_dir", type=str, default=None, help="Input directory")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory, default: os.environ['DAREA_DATA_PATH'] + 'vctk'",
    )

    args = parser.parse_args()

    main(args=args)
