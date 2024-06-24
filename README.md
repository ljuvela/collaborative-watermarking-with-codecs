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