# 🎨 Augmentarium --> One image in. A gallery of augmentations out.

Welcome to **Augmentarium** — a tiny, gorgeous web app that turns a single photo into a wall of classic image transforms, then lets you **play with them live** using sliders. 
It’s equal parts learning toy, prototyping tool, and “wow this looks cool” machine.


---

## 🎯 Why this exists (and why it’s actually useful)

* **See variety instantly.** Upload once → get a clean grid of transformations (flip, rotate, translate, crop, grayscale, RGB channels, hue/sat, brightness/contrast, invert, edges, unsharp). No knobs, no menus, just *bam* — visual variety.
* **Learn by feel.** The **Interactive Lab** adds sliders for Rotate / Translate / Zoom / Brightness / Contrast / Saturation / Hue / Blur / Unsharp / Posterize with a tiny “how it works” note. You *feel* HSV, sharpening, and blur rather than reading about them.
* **Prototype fast.** Need five looks for a thumbnail or a dataset sanity check? Generate the gallery and **Download All** as a zip. Done.
* **Zero friction.** One file input, instant thumbnail preview, elegant gallery, and friendly URLs:
  `/result/<token>` (gallery), `/lab/<token>` (lab), `/download/<token>.zip` (bundle).
* **Aesthetic by default.** Subtle glassmorphism, soft gradients, tidy labels — nothing covering the image. It just looks… nice.

---

## ✨ Features at a glance

* **Upload → Gallery:** Auto-fixes EXIF orientation and scales the long side to **768 px** for speed.
* **Interactive Lab (Sliders!):** Live preview with clear micro-explanations for each control.
* **Simple labels, no hover clutter:** Every tile is labeled (e.g., “Blue Channel Only”).
* **One-click ZIP:** Grab everything (including `original.png`) in one download.
* **Revisitable sessions:** Each run gets a short **token** so you can come back later.

---

## 🧪 What transforms are included?

**Geometry:** Horizontal/Vertical Flip • Rotate +45° • Rotate +90° • Translate (shift) • Center Crop + Resize
**Color & Tone:** Grayscale • Red/Green/Blue Channel Only • Hue +12° • Saturation +25% • Contrast +20% • Brightness +20%
**Effects:** Invert Colors • Edge Detect • Unsharp Mask
**Interactive-only extras:** Gaussian Blur • Posterize (bits per channel)

> Tip: Use a colorful portrait or outdoor photo to really see channels, hue, and edges pop.😉

---

## ⚙️ Quick Start (with `uv`)

```bash
# install exact pinned deps into .venv
uv sync

# run the app
uv run python app.py
# open http://127.0.0.1:5000
```

**Workflow**

1. Choose a file — you’ll see an immediate preview.
2. Click **Generate Augmentations** → admire the grid.
3. Click **Open Interactive Lab** to tweak sliders with live preview.
4. Use **Download All** anytime for a neat ZIP.

---

## 🗂 Folder layout

```
augmentarium/
├─ app.py
├─ pyproject.toml
├─ utils/
│  └─ image_ops.py
├─ templates/
│  ├─ index.html    # upload + instant preview
│  ├─ result.html   # labeled gallery + Download All + Lab link
│  └─ lab.html      # slider-based Interactive Lab
└─ static/
   └─ outputs/<token>/
       ├─ original.png
       ├─ 01_horizontal_flip.png
       ├─ ... (all generated PNGs)
       └─ <token>.zip
```

Routes you can bookmark:

* **Gallery:** `/result/<token>`
* **Lab:** `/lab/<token>`
* **ZIP:** `/download/<token>.zip`

---

## 🔧 Tech notes (short & sweet)

* **Stack:** Flask + Pillow + NumPy, Tailwind via CDN
* **Packaging:** `uv` with pinned versions in `pyproject.toml`
* **Reasonable defaults:** Long side 768 px, high-quality resampling, safe intensity tweaks
* **Stateless & local:** No DB, no user tracking — just files on disk inside `static/outputs/<token>/`

Pinned deps:

```
flask==3.0.3
numpy==2.0.2
pillow==10.4.0
scikit-image==0.24.0
```

*(scikit-image is pinned for future experiments; the current build runs fine with Flask + Pillow + NumPy.)*

---

## 🧩 Customize in 30 seconds

* **Add / remove transforms:** edit `AUGS` in `utils/image_ops.py`.
* **Tweak the UI:** templates live in `templates/` (Tailwind via CDN—no build step).
* **Change output size:** update `MAX_SIDE` in `image_ops.py`.

---

## 🧯 Troubleshooting

* **Looks unstyled?** Ensure you’re online (Tailwind is loaded via CDN).
* **Upload fails?** File must be JPG/PNG/WebP ≤ 25 MB.
* **Directly visiting `/lab/<token>` shows 404?** Generate once from `/` to create that token folder.

---

## 📝 License

Opensource — remix it, ship it, have fun.

---

## 🙌 Credits

Crafted with ❤️ and 👾 and a healthy respect for simple tools that teach by **showing** rather than telling.
