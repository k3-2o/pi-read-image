/**
 * OCR pipeline: ImageMagick preprocessing + native C Tesseract
 *
 * Based on the production pipeline documented in OCR.md:
 * - Tier 1: ImageMagick preprocessing (resize 250%, sharpen, contrast, grayscale, alpha, border)
 * - Tier 2: Native C Tesseract with eng_best model (if available) or default eng
 * - Tier 3: PSM 6 (uniform block), OEM 1 (LSTM only) by default
 * - Tier 4: Dictionary disabled, non-dict word penalties, ASCII whitelist for code/terminal
 * - Tier 5: Confidence extraction from TSV output + post-processing cleanup
 */

import { execFile } from "node:child_process";
import { mkdtemp, readFile, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

// --- Types ---

export interface OcrResult {
  text: string;
  confidence: number; // 0-100
  language: string;
  psm: number;
  model: string; // "best" | "fast" | "default"
}

export interface OcrOptions {
  language?: string;
  psm?: number;
  cwd?: string;
}

export interface DepsCheck {
  tesseract: boolean;
  imagemagick: boolean;
  tesseractVersion?: string;
  missingDeps: string[];
}

// --- Constants ---

// ASCII whitelist for code/terminal screenshots.
// Prevents Tesseract from hallucinating Unicode characters (emoji, smart quotes, ©, etc.)
// when the source material is pure ASCII terminal output.
const ASCII_WHITELIST =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" +
  "0123456789" +
  " .,;:!?()\"'`~-_[]{}@#$%^&*+=/\\\\|<>";

// Upscale factor for terminal screenshots.
// Tesseract was trained on 300 DPI scans; terminal screenshots are ~72 DPI.
// 250% is the sweet spot: more pixel data than 200% without excessive compute.
// For large images (>1000px), we reduce upscale to avoid ImageMagick hanging.
const UPSCALE_FACTOR = "250%";
const UPSCALE_LARGE = "150%";  // for images > 1000px in any dimension
const UPSCALE_XLARGE = "100%"; // for images > 2000px — no resize
const LARGE_IMAGE_THRESHOLD = 1000;
const XLARGE_IMAGE_THRESHOLD = 2000;

// Confidence threshold: words below this are excluded from avg
const LOW_CONFIDENCE_THRESHOLD = 10;

// --- Dependency Detection ---

let depsCache: DepsCheck | null = null;

export async function checkDependencies(): Promise<DepsCheck> {
  if (depsCache) return depsCache;

  const missing: string[] = [];
  let hasTesseract = false;
  let tesseractVersion: string | undefined;

  try {
    const { stdout } = await execFileAsync("tesseract", ["--version"], {
      timeout: 5000,
    });
    hasTesseract = true;
    tesseractVersion = stdout.split("\n")[0]?.trim();
  } catch {
    missing.push("tesseract-ocr");
  }

  let hasImageMagick = false;
  try {
    await execFileAsync("convert", ["--version"], { timeout: 5000 });
    hasImageMagick = true;
  } catch {
    try {
      await execFileAsync("magick", ["--version"], { timeout: 5000 });
      hasImageMagick = true;
    } catch {
      missing.push("imagemagick");
    }
  }

  depsCache = {
    tesseract: hasTesseract,
    imagemagick: hasImageMagick,
    tesseractVersion,
    missingDeps: missing,
  };

  return depsCache;
}

// --- Model Detection ---

let modelCache: { path?: string; label: string } | null = null;

async function findBestModel(): Promise<{ path?: string; label: string }> {
  if (modelCache) return modelCache;

  const checkPath = async (p: string): Promise<{ path?: string; label: string } | null> => {
    try {
      const s = await stat(p);
      if (s.size > 10_000_000) return { path: p, label: "best" };
      return { path: p, label: "fast" };
    } catch {
      return null;
    }
  };

  // Check TESSDATA_PREFIX env var
  const tessdataPrefix = process.env.TESSDATA_PREFIX;
  if (tessdataPrefix) {
    const bestPath = join(tessdataPrefix, "eng.traineddata");
    const result = await checkPath(bestPath);
    if (result) {
      modelCache = result;
      return modelCache;
    }
  }

  // Check ~/.local/share/tessdata
  const localPath = join(
    process.env.HOME || "~",
    ".local/share/tessdata/eng.traineddata",
  );
  const localResult = await checkPath(localPath);
  if (localResult) {
    modelCache = localResult;
    return modelCache;
  }

  // Check system tessdata paths
  const systemPaths = [
    "/usr/share/tesseract-ocr/4.00/tessdata/eng.traineddata",
    "/usr/share/tesseract-ocr/5/tessdata/eng.traineddata",
    "/usr/share/tessdata/eng.traineddata",
  ];
  for (const p of systemPaths) {
    const result = await checkPath(p);
    if (result) {
      modelCache = result;
      return modelCache;
    }
  }

  modelCache = { label: "default" };
  return modelCache;
}

// --- Convert Detection ---

let convertCache: string | null = null;

async function getConvertCommand(): Promise<string> {
  if (convertCache) return convertCache;
  try {
    await execFileAsync("convert", ["--version"], { timeout: 3000 });
    convertCache = "convert";
    return convertCache;
  } catch {
    convertCache = "magick";
    return convertCache;
  }
}

// --- Post-Processing ---

/**
 * Clean up common Tesseract OCR artifacts.
 */
function postProcess(text: string): string {
  // Fix pipe/letter confusion
  text = text.replace(/\|([A-Z])/g, "$1");
  text = text.replace(/([a-z])\|/g, "$1");

  // Fix permission-bit mangling in ls -la output
  text = text.replace(/^J(r[wxsS-]{8})/gm, "-$1");
  text = text.replace(/^d(r[a@])(x[wxsS-]{7})/gm, "drwx$2");

  // Fix $ prompt being read as ) or 5
  text = text.replace(
    /^([\s]*)[)5]\s+(ls|cd|cat|echo|grep|find|npm|git|node|python|nvim|vim|code|ssh)\b/gm,
    "$1$$ $2",
  );

  // Common symbol confusion
  text = text.replace(/©/g, "(c)");

  // Strip non-printable chars except tabs and newlines
  text = text.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F]/g, "");

  // Collapse multiple blank lines
  text = text.replace(/\n{3,}/g, "\n\n");

  return text;
}

// --- OCR Pipeline ---

/**
 * Determine adaptive upscale factor based on image dimensions.
 * Large images (e.g. generated art, high-res photos) don't need
 * aggressive upscale and can cause ImageMagick to hang.
 */
async function getAdaptiveUpscale(imagePath: string): Promise<string> {
  try {
    const convertCmd = await getConvertCommand();
    const { stdout } = await execFileAsync(convertCmd, [imagePath, "-format", "%w %h", "info:"], {
      timeout: 5000,
    });
    const [w, h] = stdout.trim().split(/\s+/).map(Number);
    if (isNaN(w) || isNaN(h)) return UPSCALE_FACTOR;
    const maxDim = Math.max(w, h);
    if (maxDim > XLARGE_IMAGE_THRESHOLD) return UPSCALE_XLARGE;
    if (maxDim > LARGE_IMAGE_THRESHOLD) return UPSCALE_LARGE;
    return UPSCALE_FACTOR;
  } catch {
    return UPSCALE_FACTOR; // can't determine size, use default
  }
}

export async function runOCR(
  imagePath: string,
  options: OcrOptions = {},
): Promise<OcrResult> {
  const language = options.language || "eng";
  const psm = options.psm ?? 6;
  const cwd = options.cwd || process.cwd();

  // Resolve absolute path
  const absPath = imagePath.startsWith("/")
    ? imagePath
    : resolve(cwd, imagePath);

  // Check file exists
  try {
    const s = await stat(absPath);
    if (!s.isFile()) {
      throw new Error(`Path is not a file: ${absPath}`);
    }
  } catch (err: any) {
    if (err.code === "ENOENT") {
      throw new Error(`Image file not found: ${absPath}`);
    }
    throw err;
  }

  // Check deps
  const deps = await checkDependencies();
  if (deps.missingDeps.length > 0) {
    throw new Error(
      `Missing required system dependencies: ${deps.missingDeps.join(", ")}.\n\n` +
        `Install with:\n` +
        `  sudo apt install tesseract-ocr imagemagick -y\n\n` +
        `For better accuracy, also install the eng_best model:\n` +
        `  wget https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata\n` +
        `  mkdir -p ~/.local/share/tessdata\n` +
        `  mv eng.traineddata ~/.local/share/tessdata/\n` +
        `  echo 'export TESSDATA_PREFIX=~/.local/share/tessdata' >> ~/.bashrc\n` +
        `  source ~/.bashrc`,
    );
  }

  // Detect model quality
  const model = await findBestModel();
  const convertCmd = await getConvertCommand();

  // Create temp directory
  const tempDir = await mkdtemp(join(tmpdir(), "pi-ocr-"));
  const preprocessed = join(tempDir, "preprocessed.png");
  const tsvOutput = join(tempDir, "output");

  try {
    // --- Tier 1: ImageMagick Preprocessing ---
    let preprocessFailed = false;
    let upscaleFactor = UPSCALE_FACTOR;
    try {
      upscaleFactor = await getAdaptiveUpscale(absPath);
    } catch {
      // use default
    }

    try {
      await execFileAsync(convertCmd, [
        absPath,
        "-resize", upscaleFactor,
        "-sharpen", "0x3",
        "-contrast-stretch", "5%",
        "-colorspace", "Gray",
        "-alpha", "off",
        "-bordercolor", "White",
        "-border", "10x10",
        preprocessed,
      ], { timeout: 60000 });
    } catch (preprocessErr: any) {
      // ImageMagick failed (timeout, OOM, or unsupported image type).
      // Fall back to OCR on the raw image without preprocessing.
      preprocessFailed = true;
    }

    const inputPath = preprocessFailed ? absPath : preprocessed;

    // --- Tier 2-4: Tesseract OCR ---
    // Always apply ASCII whitelist for the PSM modes used for code/terminal
    const useWhitelist = [3, 4, 6].includes(psm);

    const tesseractArgs: string[] = [
      inputPath,
      tsvOutput,
      "-l", language,
      "--psm", String(psm),
      "--oem", "1",
      "-c", "load_system_dawg=F",
      "-c", "load_freq_dawg=F",
      "-c", "language_model_penalty_non_dict_word=0.2",
      "-c", "language_model_penalty_non_freq_dict_word=0.15",
      "-c", "tessedit_create_tsv=1",
      "-c", "tessedit_create_txt=1",
    ];

    if (useWhitelist) {
      tesseractArgs.push("-c", `tessedit_char_whitelist=${ASCII_WHITELIST}`);
    }

    // Set TESSDATA_PREFIX for custom best model path
    const env = { ...process.env };
    let tessdataOverride = false;
    if (model.path && model.label === "best") {
      const parent = resolve(model.path, "..");
      if (
        parent !== "/usr/share/tesseract-ocr/4.00/tessdata" &&
        parent !== "/usr/share/tesseract-ocr/5/tessdata" &&
        parent !== "/usr/share/tessdata"
      ) {
        env.TESSDATA_PREFIX = parent;
        tessdataOverride = true;
      }
    }

    await execFileAsync("tesseract", tesseractArgs, {
      timeout: 60000,
      env,
      maxBuffer: 50 * 1024 * 1024,
    });

    // Read text output
    const txtPath = tsvOutput + ".txt";
    let rawText: string;
    try {
      rawText = (await readFile(txtPath, "utf8")).trim();
    } catch {
      throw new Error("Tesseract produced no text output");
    }

    const fullText = postProcess(rawText);

    // Parse TSV for word-level confidence
    const tsvPath = tsvOutput + ".tsv";
    let avgConfidence = 0;
    try {
      const tsvContent = await readFile(tsvPath, "utf8");
      const lines = tsvContent.split("\n");
      const confidences: number[] = [];

      for (const line of lines) {
        if (!line.trim()) continue;
        const cols = line.split("\t");
        const level = parseInt(cols[0], 10);
        const conf = parseFloat(cols[10]);
        const wordText = cols[11]?.trim();

        if (level === 5 && wordText && conf >= 0) {
          if (conf >= LOW_CONFIDENCE_THRESHOLD) {
            confidences.push(conf);
          }
        }
      }

      if (confidences.length > 0) {
        const sum = confidences.reduce((a, b) => a + b, 0);
        avgConfidence = Math.round((sum / confidences.length) * 10) / 10;
      }
    } catch {
      // TSV not available — confidence stays at 0
    }

    return {
      text: fullText,
      confidence: avgConfidence,
      language,
      psm,
      model: model.label + (tessdataOverride ? " (custom path)" : "") + (preprocessFailed ? " (no preprocessing)" : ""),
    };
  } catch (err: any) {
    if (err.stderr) {
      throw new Error(`Tesseract OCR failed:\n${err.stderr}`);
    }
    throw err;
  } finally {
    await rm(tempDir, { recursive: true, force: true }).catch(() => {});
  }
}

