"""
Image Classifier
Upload any image — ResNet-50 returns top-5 ImageNet predictions with confidence scores.
"""
import io
import json
import base64
import urllib.request
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights

app = Flask(__name__)

# Load model once at startup
weights = ResNet50_Weights.IMAGENET1K_V2
model   = resnet50(weights=weights)
model.eval()

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = model.to(DEVICE)
preprocess = weights.transforms()

IMAGENET_LABELS_URL = (
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
)
_LABELS_CACHE = Path(".imagenet_labels.txt")

def get_labels() -> list[str]:
    if not _LABELS_CACHE.exists():
        urllib.request.urlretrieve(IMAGENET_LABELS_URL, _LABELS_CACHE)
    return _LABELS_CACHE.read_text().strip().splitlines()

LABELS = get_labels()


def classify(img: Image.Image) -> list[dict]:
    tensor = preprocess(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    top5  = probs.topk(5)
    return [
        {"label": LABELS[idx], "confidence": round(prob.item() * 100, 2)}
        for prob, idx in zip(top5.values, top5.indices)
    ]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/classify", methods=["POST"])
def classify_route():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file  = request.files["image"]
    image = Image.open(io.BytesIO(file.read()))

    # Thumbnail for response preview
    thumb = image.copy()
    thumb.thumbnail((300, 300))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=85)
    thumb_b64 = base64.b64encode(buf.getvalue()).decode()

    predictions = classify(image)

    return jsonify({"predictions": predictions, "thumb": thumb_b64})


if __name__ == "__main__":
    print(f"Running on device: {DEVICE}")
    app.run(debug=True, port=5004)
