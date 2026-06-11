"""
Image Classifier
Tries to forward inference to the necron GPU worker first (fast).
Falls back to CPU inference on the VPS using torchvision (slower but always works).
"""
import io
import json
import time
import urllib.request
from flask import Flask, render_template, request, jsonify
import requests as http

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

NECRON          = "http://100.72.210.90:15100"
CONNECT_TIMEOUT = 4
READ_TIMEOUT    = 120

# ── CPU fallback (lazy-loaded on first use) ───────────────────────────
_cpu_models = {}

def _load_cpu_model(name: str):
    """Load a torchvision model onto CPU (cached after first load)."""
    if name in _cpu_models:
        return _cpu_models[name]

    import torch
    import torchvision.models as models
    import torchvision.transforms as T

    model_map = {
        "resnet50":       (models.resnet50,       models.ResNet50_Weights.DEFAULT),
        "efficientnet_b0":(models.efficientnet_b0,models.EfficientNet_B0_Weights.DEFAULT),
        "mobilenet_v3":   (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT),
    }
    # The UI sends short names; map them onto the torchvision keys
    aliases = {"efficientnet": "efficientnet_b0", "mobilenet": "mobilenet_v3"}
    name = aliases.get(name, name)
    if name not in model_map:
        name = "resnet50"

    factory, weights = model_map[name]
    model = factory(weights=weights)
    model.eval()
    _cpu_models[name] = (model, weights)
    return model, weights


def _classify_cpu(raw_bytes: bytes, model_name: str, top_n: int) -> dict:
    """Run classification on CPU and return a result dict."""
    import torch
    from PIL import Image

    model, weights = _load_cpu_model(model_name)
    transform = weights.transforms()

    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
    elapsed = time.perf_counter() - t0

    # Build label list from weights metadata
    categories = weights.meta["categories"]
    top_indices = probs.argsort(descending=True)[:top_n].tolist()
    predictions = [
        {
            "rank":       i + 1,
            "label":      categories[idx].replace("_", " "),
            "confidence": round(float(probs[idx].item()) * 100, 2),
        }
        for i, idx in enumerate(top_indices)
    ]

    return {
        "model":        model_name,
        "predictions":  predictions,
        "inference_ms": round(elapsed * 1000, 1),
        "device":       "cpu",
    }


# ── Routes ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/classify", methods=["POST"])
def classify_route():
    # File upload path
    if "image" in request.files:
        file       = request.files["image"]
        model_name = request.form.get("model", "resnet50")
        top_n      = int(request.form.get("top_n", "10"))
        raw        = file.read()

        # Try necron GPU first
        try:
            resp = http.post(
                f"{NECRON}/classify",
                files={"image": (file.filename, raw, file.content_type)},
                data={"model": model_name, "top_n": str(top_n)},
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            )
            return (resp.content, resp.status_code, {"Content-Type": "application/json"})
        except http.exceptions.RequestException:
            pass

        # CPU fallback
        try:
            result = _classify_cpu(raw, model_name, top_n)
            result["cpu_fallback"] = True
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": f"Classification failed: {e}", "gpu_offline": True}), 503

    # URL path
    if request.is_json and request.json.get("url"):
        data       = request.json
        url        = data["url"]
        model_name = data.get("model", "resnet50")
        top_n      = int(data.get("top_n", 10))

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=8) as r:
                raw = r.read()
        except Exception as e:
            return jsonify({"error": f"Could not fetch URL: {e}"}), 400

        # Try necron GPU first
        try:
            resp = http.post(
                f"{NECRON}/classify",
                files={"image": ("url_image.jpg", raw, "image/jpeg")},
                data={"model": model_name, "top_n": str(top_n)},
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            )
            return (resp.content, resp.status_code, {"Content-Type": "application/json"})
        except http.exceptions.RequestException:
            pass

        # CPU fallback
        try:
            result = _classify_cpu(raw, model_name, top_n)
            result["cpu_fallback"] = True
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": f"Classification failed: {e}", "gpu_offline": True}), 503

    return jsonify({"error": "No image provided"}), 400



@app.after_request
def _no_html_cache(resp):
    # Browsers heuristically cache HTML served without Cache-Control, which
    # leaves visitors on stale pages after a deploy. Force revalidation.
    if resp.mimetype == "text/html":
        resp.headers["Cache-Control"] = "no-cache"
    return resp

if __name__ == "__main__":
    app.run(debug=True, port=5004)
