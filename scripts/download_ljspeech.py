import os
import argparse
import torchaudio

# Download LJSPEECH dataset with torchaudio 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download LJSPEECH dataset')
    parser.add_argument('--target_dir', type=str, default='data/ljspeech')
    args = parser.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)

    torchaudio.datasets.LJSPEECH(root=args.target_dir, download=True)
    print('LJSPEECH dataset downloaded to {}'.format(args.target_dir))