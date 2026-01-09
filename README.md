# Pixel Art Diffusion

This repository contains code for training pixel-art diffusion models.


## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/JuraH32/pixel-art-diffusion.git
cd pixel-art-diffusion
uv sync # or: pip install -r requirements.txt

# 2. Configure Kaggle API (one-time setup)
#    - Go to https://www.kaggle.com/settings → Create New Token
#    - Place downloaded kaggle.json in ~/.kaggle/
#    - chmod 600 ~/.kaggle/kaggle.json
#    Detailed instructions: https://www.kaggle.com/docs/api

# 3. Download dataset
python setup_data.py

# 4. Train
python train.py --data_path ./data/pixel_art_dataset
```

## Dataset

The model trains on the [Pixel Art dataset](https://www.kaggle.com/datasets/ebrahimelgazar/pixel-art) from Kaggle:
- 89,400 16x16 pixel art images
- 5 classes: characters, creatures, food, items, character sideview

## Usage

### Training

Basic training:
```bash
python train.py --data_path ./data/pixel_art_dataset --image_size 16 --batch_size 128
```

Higher-quality model (more memory/compute):
```bash
python train.py --data_path ./data/pixel_art_dataset --base_channels 128 --epochs 200
```

All training arguments:
| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | (required) | Path to dataset |
| `--image_size` | 16 | Image size |
| `--batch_size` | 64 | Batch size |
| `--epochs` | 150 | Number of epochs |
| `--lr` | 1e-4 | Learning rate |
| `--save_every` | 5 | Save checkpoint every N epochs |
| `--base_channels` | 128 | Model width |
| `--log_dir` | ./runs | TensorBoard log directory |
| `--checkpoint_dir` | ./checkpoints | Checkpoint save directory |

### Monitoring

View training metrics with TensorBoard:
```bash
tensorboard --logdir ./runs --port 6006
```
Open http://localhost:6006 in your browser.

### Generation

After training, use the notebook `visualize_data.ipynb` to generate samples, or:

```python
from model import ScalableUNet, PixelDiffusion
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ScalableUNet(num_classes=5, img_size=16, base_c=128, t_dim=256).to(device)
model.load_state_dict(torch.load("checkpoints/model_e150.pt", map_location=device))

diffusion = PixelDiffusion(model, image_size=16, device=device)

# Generate 4 samples of class 1 (characters)
labels = torch.tensor([1, 1, 1, 1]).to(device)
samples = diffusion.sample(labels=labels, cfg_scale=7.0)
```

## Project Structure

```
pixel-art-diffusion/
├── setup_data.py      # Download dataset from Kaggle
├── dataloader.py      # Dataset and data loading
├── model.py           # UNet architecture and diffusion logic
├── train.py           # Training script
├── visualize_data.ipynb  # Visualization and generation notebook
├── data/              # Dataset directory (created by setup_data.py)
│   └── pixel_art_dataset/
│       ├── sprites.npy
│       └── sprites_labels.npy
├── checkpoints/       # Saved model weights (created during training)
└── runs/              # TensorBoard logs (created during training)
```

## Requirements

- Python 3.8+
- PyTorch
- CUDA (optional, for GPU training)
- Kaggle account (for dataset download)