# Image Classifier

Upload any image and a pretrained **ResNet-50** model classifies it from 1,000 ImageNet categories. Returns top-5 predictions with confidence scores.

Auto-detects GPU — runs on CPU by default, CUDA on necron.

## Quick Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
# → http://localhost:5004
```

The model weights (~100 MB) download automatically on first run via torchvision.

## Running on necron (RTX 5080)

```bash
ssh necron   # Tailscale
cd ~/projects/image-classifier
source .venv/bin/activate && python app.py
# CUDA detected automatically — inference is near-instant
```

## Tech Stack

Python · Flask · PyTorch · torchvision ResNet-50 · Pillow
