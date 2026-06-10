# Image Classifier

Classify any image with ImageNet-pretrained CNNs, with automatic GPU-to-CPU failover.

**Live demo:** [classifier.spbdatascience.org](https://classifier.spbdatascience.org)

## Features

- Three models: ResNet-50, EfficientNet-B0, MobileNet-V3
- File upload or direct image URL
- Top-N predictions (up to 20) with animated confidence bars
- Tries the club's RTX 5080 worker first; if it is offline, inference falls back to CPU torchvision on the VPS so the demo never goes dark

## How it works

The Flask app forwards uploads to a GPU worker over Tailscale. On connection failure it lazily loads the requested torchvision model on CPU (cached after first load), applies the model's own preprocessing transforms, and serves the softmax top-N locally. The response includes which device served the request.

## Stack

Python, Flask, PyTorch, torchvision, Pillow

## Local development

```bash
pip install flask requests torch torchvision pillow
python app.py
```
