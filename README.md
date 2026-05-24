# pi-read-image

**OCR image-to-text tool for [pi](https://pi.dev)** — extracts text from screenshots, terminal output, and code images using native C Tesseract with a production-grade ImageMagick preprocessing pipeline.

> **Accuracy:** ~82-90% on terminal screenshots (vs ~48% for tesseract.js)

## What it does

When you ask pi to read an image (`.png`, `.jpg`, `.webp`, `.bmp`, `.tiff`, `.gif`), the built-in `read` tool may return *"Current model does not support images"* if the active LLM lacks vision capabilities. This extension provides a **local OCR fallback** that:

1. Preprocesses the image with ImageMagick (adaptive upscale, sharpen, contrast, grayscale, border)
2. Runs native C Tesseract with LSTM engine (OEM 1)
3. Extracts word-level confidence from TSV output
4. Cleans up common OCR artifacts (pipe/letter confusion, permission-bit mangling, smart quotes)

If OCR confidence is below 50% **and** the model supports images, the extension automatically falls back to sending the image directly to the model's vision capability — so you get the best of both worlds.

## Installation

### System dependencies

You need native C Tesseract and ImageMagick installed on your system:

```bash
# Debian/Ubuntu
sudo apt install tesseract-ocr imagemagick -y

# Fedora
sudo dnf install tesseract imagemagick -y

# macOS
brew install tesseract imagemagick
```

### Pi package

```bash
pi install git:github.com/k3-2o/pi-read-image
```

Verify it's loaded:

```bash
pi --list-tools | grep read_image
```

You should see `read_image` listed among the available tools.

### Optional: eng_best model (~20% accuracy boost)

The default Tesseract `eng` model is a fast, compact model. For significantly better accuracy on screenshots and terminal output, install the **eng_best** model:

```bash
wget https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata
mkdir -p ~/.local/share/tessdata
mv eng.traineddata ~/.local/share/tessdata/
echo 'export TESSDATA_PREFIX=~/.local/share/tessdata' >> ~/.bashrc
source ~/.bashrc
```

The extension auto-detects the best model. If it finds a traineddata file larger than 10MB, it uses it as the "best" engine.

## Usage

The LLM uses `read_image` **automatically** when it encounters an image file the model cannot see. You don't need to call it manually — just ask pi to read an image like you normally would.

### Manual invocation (for advanced users)

You can also call the tool directly through pi's tool system. Parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `string` | — | Path to the image file |
| `language` | `string` | `"eng"` | OCR language code (ISO 639-3) |
| `psm` | `number` | `6` | Page Segmentation Mode (see table below) |

### PSM Mode Reference

| PSM | Description | Best for |
|-----|-------------|----------|
| `3` | Fully automatic page segmentation | Mixed content, UI screenshots |
| `4` | Single column of text of variable sizes | Articles, documents |
| `6` | **Uniform block of text (default)** | Code screenshots, terminal output, logs |
| `7` | Treat image as single text line | One-liners, status messages |
| `11` | Sparse text without specific order | Random text placement |
| `13` | Raw line (bypasses Tesseract layout analysis) | Debugging, raw output |

### Confidence-based fallback

- **Confidence ≥ 80%**: Results displayed normally (green indicator)
- **Confidence 50-79%**: Results shown with warning (yellow indicator)
- **Confidence < 50%**: If model supports images → falls back to direct model vision. If not → results shown with error indicator and a warning message
- **Confidence < 30%**: The image may not contain readable text

## How it works

```
Image file
    │
    ▼
ImageMagick preprocessing
  ├─ Adaptive upscale (250% / 150% / 100% based on dimensions)
  ├─ Sharpen (0x3)
  ├─ Contrast stretch (5%)
  ├─ Grayscale conversion
  ├─ Alpha channel removal
  └─ White border (10px)
    │
    ▼
Tesseract OCR (OEM 1 — LSTM only)
  ├─ ASCII whitelist for code/terminal
  ├─ Dictionary penalties disabled
  └─ Non-dict word penalties increased
    │
    ▼
Post-processing
  ├─ Fix pipe/letter confusion
  ├─ Fix permission-bit mangling
  ├─ Fix $ prompt detection
  └─ Strip non-printable characters
    │
    ▼
Confidence < 50% and model has vision?
  ├─ Yes → Fall back to model vision (attached image)
  └─ No → Return OCR text with confidence indicator
```

## File extension support

| Extension | MIME type |
|-----------|-----------|
| `.png` | `image/png` |
| `.jpg`, `.jpeg` | `image/jpeg` |
| `.webp` | `image/webp` |
| `.gif` | `image/gif` |
| `.bmp` | `image/bmp` |
| `.tiff`, `.tif` | `image/tiff` |

## Credits

Built by [k3-2o](https://github.com/k3-2o) for the pi ecosystem.

## License

MIT — see [LICENSE](LICENSE).
