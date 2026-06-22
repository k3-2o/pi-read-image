# pi-read-image

OCR image-to-text tool for [Pi](https://pi.dev) — extracts text from screenshots, terminal output, and code images using Tesseract + ImageMagick.

## What it does

When Pi encounters an image the model can't see, it OCRs it locally. Preprocesses with ImageMagick (adaptive upscale, sharpen, contrast, grayscale), runs Tesseract with LSTM engine, extracts word-level confidence, and cleans up artifacts. If confidence is low and the model has vision, the image is sent directly to the model instead.

Pass an array of paths to OCR multiple images in one call — they process concurrently and one failure won't waste the batch.

## Requirements

- `tesseract-ocr`
- `imagemagick`

```bash
# Debian/Ubuntu
sudo apt install tesseract-ocr imagemagick -y
# macOS
brew install tesseract imagemagick
```

## Install

```bash
pi install git:github.com/k3-2o/pi-read-image
```

## Usage

The model uses `read_image` automatically when it can't see an image. Parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `string` \| `string[]` | — | Path to image file, or array of paths for batch OCR |
| `language` | `string` | `"eng"` | OCR language code (ISO 639-3) |
| `psm` | `number` | `6` | Page Segmentation Mode (3=auto, 4=single column, 6=block of text, 7=single line, 11=sparse, 13=raw) |

## License

MIT
