import os
import io
import uuid
import zipfile
from PIL import Image
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, send_file, abort, jsonify
from werkzeug.utils import secure_filename

from utils.image_ops import (
    load_and_prepare,
    generate_augmentations_with_verdicts,
)

APP_NAME = "Augmentarium"
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
OUTPUT_BASE = os.path.join(STATIC_DIR, "outputs")

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  # dev-friendly cache


def _new_token():
    return uuid.uuid4().hex[:12]


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _allowed(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXT


def _zip_dir(dir_path: str) -> bytes:
    memf = io.BytesIO()
    with zipfile.ZipFile(memf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(dir_path):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, dir_path)
                    zf.write(full, arcname=rel)
    memf.seek(0)
    return memf.read()


def _orig_path_for(token: str) -> str:
    return os.path.join(OUTPUT_BASE, token, "original.png")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", app_name=APP_NAME)


@app.route("/augment", methods=["POST"])
def augment():
    if "image" not in request.files:
        return redirect(url_for("index"))
    file = request.files["image"]
    if file.filename == "":
        return redirect(url_for("index"))

    if not _allowed(file.filename):
        return render_template(
            "index.html",
            app_name=APP_NAME,
            error="Unsupported file type. Please upload JPG/PNG/WebP.",
        )

    token = _new_token()
    out_dir = os.path.join(OUTPUT_BASE, token)
    _ensure_dir(out_dir)

    # Save original
    safe_name = secure_filename(file.filename)
    original_name = "original.png"  # normalize to PNG for consistency
    original_path = os.path.join(out_dir, original_name)
    img_pil = load_and_prepare(file.stream)  # handles EXIF, resizing, RGB
    img_pil.save(original_path)

    # Generate 12 augmentations + verdicts
    items = generate_augmentations_with_verdicts(img_pil, out_dir, token)

    # Create zip bytes for download
    zip_bytes = _zip_dir(out_dir)
    zip_token = f"{token}.zip"
    zip_path = os.path.join(out_dir, zip_token)
    with open(zip_path, "wb") as f:
        f.write(zip_bytes)

    return render_template(
        "result.html",
        app_name=APP_NAME,
        token=token,
        original_filename=original_name,
        items=items,
        zipfile_name=zip_token,
    )


@app.route("/download/<token>.zip", methods=["GET"])
def download_zip(token):
    # serve the zip built under static/outputs/<token>/<token>.zip
    zpath = os.path.join(OUTPUT_BASE, token, f"{token}.zip")
    if not os.path.exists(zpath):
        abort(404)
    return send_file(
        zpath,
        as_attachment=True,
        download_name=f"augmentarium_{token}.zip",
        mimetype="application/zip",
    )


@app.after_request
def add_header(resp):
    # no aggressive caching in dev
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/lab/<token>", methods=["GET"])
def lab(token):
    orig = _orig_path_for(token)
    if not os.path.exists(orig):
        abort(404)
    return render_template(
        "lab.html",
        app_name=APP_NAME,
        token=token,
        original_filename="original.png",
    )


@app.route("/api/preview/<token>", methods=["POST"])
def api_preview(token):
    orig = _orig_path_for(token)
    if not os.path.exists(orig):
        abort(404)
    data = request.get_json(force=True, silent=True) or {}
    key = (data.get("key") or "").strip()
    params = data.get("params") or {}
    try:
        img = Image.open(orig).convert("RGB")
        from utils.image_ops import parametric_apply  # lazy import
        out = parametric_apply(img, key, params)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    buf = io.BytesIO()
    out.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/result/<token>", methods=["GET"])
def result(token):
    out_dir = os.path.join(OUTPUT_BASE, token)
    original_path = os.path.join(out_dir, "original.png")
    if not os.path.exists(original_path):
        abort(404)

    # Rebuild items from saved files using AUGS naming pattern
    from utils.image_ops import AUGS
    items = []
    for idx, (label, _) in enumerate(AUGS, start=1):
        fname = f"{idx:02d}_{label.lower().replace(' ', '_').replace('°','deg').replace('+','plus')}.png"
        fpath = os.path.join(out_dir, fname)
        if os.path.exists(fpath):
            items.append({"name": label, "filename": fname, "idx": idx})

    # zip might already exist (created during /augment POST). It's okay if it doesn't.
    zipfile_name = f"{token}.zip" if os.path.exists(os.path.join(out_dir, f"{token}.zip")) else None

    return render_template(
        "result.html",
        app_name=APP_NAME,
        token=token,
        original_filename="original.png",
        items=items,
        zipfile_name=zipfile_name or f"{token}.zip",
    )


if __name__ == "__main__":
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    # 0.0.0.0 so it works on EC2/public
    app.run(host="0.0.0.0", port=5000, debug=False)
