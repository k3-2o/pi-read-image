/**
 * read_image — OCR tool for pi
 *
 * Extracts text from images (screenshots, code, terminal output, logs)
 * using native C Tesseract with a production-grade preprocessing pipeline.
 *
 * This tool is the fallback for models that lack vision capabilities.
 * When read() returns "Current model does not support images", the model
 * should use read_image to extract text from the image instead.
 *
 * Accuracy: ~82-90% on terminal screenshots (vs ~48% for tesseract.js)
 *
 * Requirements:
 *   sudo apt install tesseract-ocr imagemagick    (Linux)
 *   brew install tesseract imagemagick             (macOS)
 *
 * Optional: eng_best model for ~20% accuracy boost (auto-detected if installed)
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import {
  DEFAULT_MAX_BYTES,
  DEFAULT_MAX_LINES,
  formatSize,
  truncateHead,
} from "@earendil-works/pi-coding-agent";
import { Text } from "@earendil-works/pi-tui";
import { readFile, stat } from "node:fs/promises";
import { extname, resolve } from "node:path";
import { Type } from "typebox";

import { checkDependencies, runOCR, type OcrResult } from "./ocr";

// --- Schema ---

const ReadImageParams = Type.Object({
  path: Type.Union([Type.String(), Type.Array(Type.String())], {
    description:
      "Path to the image file, or array of multiple image paths to batch OCR (png, jpg, webp, etc.)",
  }),
  language: Type.Optional(
    Type.String({
      description: "OCR language code. Default: 'eng'. Use 'eng' for English, or ISO 639-3 codes for others (e.g., 'fra' for French). Requires traineddata to be installed.",
    }),
  ),
  psm: Type.Optional(
    Type.Number({
      description:
        "Page Segmentation Mode. 3=auto, 4=single column, 6=uniform block of text (default, best for code/terminal), 7=single line, 11=sparse text, 13=raw line. Default: 6.",
    }),
  ),
});

// --- Rendering types ---

interface ReadImageDetails {
  path: string;
  confidence: number;
  language: string;
  psm: number;
  model: string;
  truncated: boolean;
  totalLines: number;
  totalBytes: number;
  /** Which path handled the image: "ocr" (Tesseract ≥ 50%), "vision_fallback" (OCR < 50%, sent to model), "ocr_no_vision" (OCR < 50%, model lacks vision) */
  approach: "ocr" | "vision_fallback" | "ocr_no_vision";
  /** When >1, this is a batch result. Details reflect the last image; confidence is averaged. */
  imageCount?: number;
}

// --- Main Extension ---

export default function (pi: ExtensionAPI) {
  let shownModelSuggestion = false;
  let depsChecked = false;
  let depsOk = false;
  let depsErrorMessage = "";

  /** Resolve a path (with @ prefix convention) to an absolute path. */
  function resolveImagePath(raw: string, cwd: string | undefined): string {
    const cleaned = raw.replace(/^@/, "");
    return cleaned.startsWith("/")
      ? cleaned
      : resolve(cwd || process.cwd(), cleaned);
  }

  /** OCR a single image and return its output text + details. */
  async function ocrSingle(
    absPath: string,
    language: string,
    psm: number,
    cwd: string | undefined,
  ): Promise<{ text: string; details: ReadImageDetails }> {
    let fileSize = 0;
    try {
      const s = await stat(absPath);
      fileSize = s.size;
      if (fileSize > 50 * 1024 * 1024) {
        throw new Error(`Image too large (${formatSize(fileSize)}). Maximum: 50MB.`);
      }
    } catch (err: any) {
      if (err.code === "ENOENT") {
        throw new Error(`Image file not found: ${absPath}`);
      }
      throw err;
    }

    let result: OcrResult;
    try {
      result = await runOCR(absPath, { language, psm, cwd });
    } catch (err: any) {
      throw new Error(`OCR failed: ${err.message}`);
    }

    if (
      !shownModelSuggestion &&
      (result.model === "default" || result.model === "fast")
    ) {
      shownModelSuggestion = true;
    }

    const header = [
      `[OCR Result: ${absPath}]`,
      `Confidence: ${result.confidence}%`,
      `Language: ${result.language} (PSM ${result.psm})`,
      `Engine: ${result.model} model`,
    ];

    if (result.model !== "best") {
      header.push(
        `Note: Install eng_best model for ~20% better accuracy: ` +
          `wget https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata -O ~/.local/share/tessdata/eng.traineddata`,
      );
    }

    header.push("", "--- Extracted Text ---", "");

    const headerText = header.join("\n");
    const fullOutput = headerText + result.text;

    const truncation = truncateHead(fullOutput, {
      maxLines: DEFAULT_MAX_LINES,
      maxBytes: DEFAULT_MAX_BYTES,
    });

    let outputText = truncation.content;
    const details: ReadImageDetails = {
      path: absPath,
      confidence: result.confidence,
      language: result.language,
      psm: result.psm,
      model: result.model,
      truncated: truncation.truncated,
      totalLines: truncation.totalLines,
      totalBytes: truncation.totalBytes,
      approach: result.confidence >= 50 ? "ocr" : "ocr_no_vision",
    };

    if (truncation.truncated) {
      const truncatedLines = truncation.totalLines - truncation.outputLines;
      const truncatedBytes = truncation.totalBytes - truncation.outputBytes;
      outputText +=
        `\n\n[Output truncated: showing ${truncation.outputLines} of ${truncation.totalLines} lines ` +
        `(${formatSize(truncation.outputBytes)} of ${formatSize(truncation.totalBytes)}). ` +
        `${truncatedLines} lines (${formatSize(truncatedBytes)}) omitted.]`;
    }

    return { text: outputText, details };
  }

  pi.registerTool({
    name: "read_image",
    label: "Read Image (OCR)",
    description: `Extract text from one or more images using OCR (Tesseract with preprocessing pipeline). Use this when the model cannot see images directly — for example, when read() returns an error about missing vision support, or when the user asks about content in a screenshot or image file. Pass a single path for one image, or an array of paths to batch multiple images in one call — they OCR in parallel and one failure won't waste the batch. Supports png, jpg, webp, and other common image formats. Output truncated to ${DEFAULT_MAX_LINES} lines or ${formatSize(DEFAULT_MAX_BYTES)}.`,

    promptSnippet: "Extract text from images via OCR",

    promptGuidelines: [
      "Always use read_image first for any image file — it runs local OCR (free, fast). The tool will automatically fall back to model vision if OCR confidence is low and the model supports images.",
      "Use read_image for file extensions like .png, .jpg, .jpeg, .webp, .bmp, .tiff, .gif.",
      "For code screenshots and terminal output, use the default PSM=6. For mixed-content images with UI elements, try PSM=3 or PSM=4. For a single line of text, use PSM=7.",
      "If read_image returns confidence below 50%, warn the user that OCR may be unreliable and ask them to verify the output. If confidence is below 30%, the image may not contain readable text.",
      "If read_image fails with a dependency error, tell the user to install tesseract-ocr and imagemagick and try again.",
      "To OCR multiple images in one call, pass an array of paths (e.g. [\"a.png\", \"b.png\"]). Vision fallback only works for single-image calls — if a batch image has low confidence, call read_image again with just that path to use model vision.",
    ],

    parameters: ReadImageParams,

    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      if (!depsChecked) {
        const deps = await checkDependencies();
        depsChecked = true;
        if (deps.missingDeps.length > 0) {
          depsOk = false;
          depsErrorMessage = buildDepsError(deps.missingDeps);
        } else {
          depsOk = true;
        }
      }
      if (!depsOk) throw new Error(depsErrorMessage);

      const language = params.language || "eng";
      const psm = params.psm ?? 6;
      const rawPaths = Array.isArray(params.path) ? params.path : [params.path];
      const cwd = ctx.cwd;

      // --- Single-image fast path (backward compatible) ---
      if (rawPaths.length === 1) {
        const absPath = resolveImagePath(rawPaths[0], cwd);
        let { text, details } = await ocrSingle(absPath, language, psm, cwd);

        const modelSupportsVision = ctx?.model?.input?.includes?.("image");
        if (details.confidence < 50 && modelSupportsVision) {
          let buffer: Buffer;
          try {
            buffer = await readFile(absPath);
          } catch (err: any) {
            throw new Error(`Vision fallback failed: cannot read image file: ${err.message}`);
          }
          const base64 = buffer.toString("base64");
          const mimeType = detectImageMimeType(absPath);
          const fallbackNote = [
            `[OCR fallback] Confidence too low (${details.confidence}%). Falling back to direct model vision.`,
            `The image has been attached below for the model to read directly.`,
          ].join("\n");

          return {
            content: [
              { type: "text", text: fallbackNote },
              { type: "image", data: base64, mimeType },
            ],
            details: { ...details, approach: "vision_fallback" as const },
          };
        }

        if (details.confidence < 50) {
          text += `\n\n⚠ Low confidence (${details.confidence}%). Results may be unreliable. Consider verifying the output manually or trying a different PSM mode.`;
        }

        return { content: [{ type: "text", text }], details };
      }

      // --- Batch path (multiple images, parallel + partial) ---
      const absPaths = rawPaths.map((p) => resolveImagePath(p, cwd));

      const results = await Promise.allSettled(
        absPaths.map((p) => ocrSingle(p, language, psm, cwd)),
      );

      const outputs: string[] = [];
      const errors: string[] = [];
      let totalConfidence = 0;
      let successCount = 0;
      let lastDetails: ReadImageDetails | null = null;

      for (let i = 0; i < results.length; i++) {
        const r = results[i];
        if (r.status === "fulfilled") {
          outputs.push(r.value.text);
          totalConfidence += r.value.details.confidence;
          successCount++;
          lastDetails = r.value.details;
        } else {
          const reason = r.reason?.message ?? String(r.reason ?? "unknown error");
          errors.push(`[${absPaths[i]}] ${reason}`);
        }
      }

      const avgConfidence =
        successCount > 0
          ? Math.round((totalConfidence / successCount) * 10) / 10
          : 0;

      const batchHeaderParts = [
        `[Batch OCR: ${rawPaths.length} images]`,
      ];
      if (successCount > 0) {
        batchHeaderParts.push(
          `Average Confidence: ${avgConfidence}% (${successCount}/${rawPaths.length} succeeded)`,
        );
      }
      batchHeaderParts.push(`Language: ${language} (PSM ${psm})`);
      if (errors.length > 0) {
        batchHeaderParts.push("", "Failed:");
        for (const e of errors) batchHeaderParts.push(`  ⚠ ${e}`);
      }
      batchHeaderParts.push("", "---", "");

      const combinedText =
        batchHeaderParts.join("\n") + outputs.join("\n\n---\n\n");

      const truncation = truncateHead(combinedText, {
        maxLines: DEFAULT_MAX_LINES,
        maxBytes: DEFAULT_MAX_BYTES,
      });

      let finalText = truncation.content;

      if (truncation.truncated) {
        finalText += `\n\n[Output truncated: showing ${truncation.outputLines} of ${truncation.totalLines} lines (${formatSize(truncation.outputBytes)} of ${formatSize(truncation.totalBytes)}).]`;
      }

      return {
        content: [{ type: "text", text: finalText }],
        details: {
          ...(lastDetails ?? {
            path: "",
            confidence: avgConfidence,
            language,
            psm,
            model: "",
            truncated: truncation.truncated,
            totalLines: truncation.totalLines,
            totalBytes: truncation.totalBytes,
            approach: "ocr" as const,
          }),
          confidence: avgConfidence,
          imageCount: rawPaths.length,
        },
      };
    },

    renderCall(args, theme, _context) {
      const paths = Array.isArray(args.path) ? args.path : [args.path || "(image)"];
      let text = theme.fg("toolTitle", theme.bold("read_image "));
      if (paths.length === 1) {
        text += theme.fg("dim", paths[0]);
      } else {
        text += theme.fg("dim", `${paths.length} images`);
      }
      if (args.language && args.language !== "eng") {
        text += theme.fg("muted", ` lang:${args.language}`);
      }
      if (args.psm && args.psm !== 6) {
        text += theme.fg("muted", ` psm:${args.psm}`);
      }
      return new Text(text, 0, 0);
    },

    renderResult(result, { expanded, isPartial }, theme, _context) {
      if (isPartial) {
        return new Text(theme.fg("warning", "Running OCR..."), 0, 0);
      }

      const details = result.details as ReadImageDetails | undefined;
      if (!details || details.confidence === undefined) {
        return new Text(theme.fg("error", "OCR failed"), 0, 0);
      }

      const isBatch = (details.imageCount ?? 1) > 1;

      if (isBatch) {
        let display = theme.fg("toolTitle", theme.bold("Batch OCR"));
        display += theme.fg("dim", ` · ${details.imageCount} images`);
        const confColor =
          details.confidence >= 80
            ? "success"
            : details.confidence >= 50
              ? "warning"
              : "error";
        display += ` ${theme.fg(confColor, `${details.confidence}% avg`)}`;
        display += theme.fg("dim", ` · ${details.language} · psm ${details.psm}`);
        if (details.truncated) display += theme.fg("warning", " · TRUNCATED");

        if (expanded) {
          const content = result.content[0];
          if (content?.type === "text") {
            for (const line of content.text.split("\n").slice(0, 10)) {
              display += `\n${theme.fg("dim", line)}`;
            }
          }
        }
        return new Text(display, 0, 0);
      }

      const confColor =
        details.confidence >= 80
          ? "success"
          : details.confidence >= 50
            ? "warning"
            : "error";

      let display = theme.fg(confColor, `${details.confidence}% confidence`);

      if (details.approach === "vision_fallback") {
        display += theme.fg("info", " → model vision");
      } else if (details.approach === "ocr_no_vision") {
        display += theme.fg("error", " ⚠ model lacks vision");
      }

      display += theme.fg("dim", ` · ${details.language} · psm ${details.psm} · ${details.model}`);

      if (details.truncated) {
        display += theme.fg("warning", " · TRUNCATED");
      }

      if (expanded) {
        const content = result.content[0];
        if (content?.type === "text") {
          const lines = content.text.split("\n");
          const textStart = lines.findIndex((l) => l === "--- Extracted Text ---");
          const contentLines = textStart >= 0 ? lines.slice(textStart + 1).slice(0, 15) : lines.slice(0, 15);
          for (const line of contentLines) {
            display += `\n${theme.fg("dim", line)}`;
          }
        }
      }

      return new Text(display, 0, 0);
    },
  });
}

// --- Helpers ---

/** Map file extension to image MIME type (same set supported by pi's read tool). */
function detectImageMimeType(filePath: string): string {
  const ext = extname(filePath).toLowerCase();
  const map: Record<string, string> = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
  };
  return map[ext] || "image/png";
}

function buildDepsError(missing: string[]): string {
  const parts = [
    `OCR tool cannot run: missing system dependencies (${missing.join(", ")}).`,
    "",
    "Install them with one of the following:",
    "",
  ];

  if (process.platform === "linux") {
    parts.push("  # Debian/Ubuntu");
    parts.push("  sudo apt install tesseract-ocr imagemagick -y");
    parts.push("");
    parts.push("  # Fedora");
    parts.push("  sudo dnf install tesseract imagemagick -y");
  } else if (process.platform === "darwin") {
    parts.push("  brew install tesseract imagemagick");
  } else {
    parts.push("  Install tesseract-ocr and imagemagick for your platform.");
  }

  parts.push("");
  parts.push("For better accuracy (~20% boost), also download the eng_best model:");
  parts.push("  wget https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata");
  parts.push("  mkdir -p ~/.local/share/tessdata");
  parts.push("  mv eng.traineddata ~/.local/share/tessdata/");
  parts.push("  echo 'export TESSDATA_PREFIX=~/.local/share/tessdata' >> ~/.bashrc");

  return parts.join("\n");
}
