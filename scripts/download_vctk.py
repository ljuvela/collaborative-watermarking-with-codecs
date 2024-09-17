import os
import argparse
import torchaudio


# Download VCTK dataset with torchaudio 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download VCTK dataset')
    parser.add_argument('--target_dir', type=str, default='data/vctk')
    args = parser.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)

    torchaudio.datasets.VCTK_092(root=args.target_dir, download=True)
    print('VCTK dataset downloaded to {}'.format(args.target_dir))