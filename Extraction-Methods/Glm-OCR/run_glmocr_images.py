#!/usr/bin/env python3
"""
run_glmocr_images.py

Input: a single image file OR a directory containing images.
Output:
  outputs_image/
    <image_name>/
      <image_name>.<ext>              (optional copy)
      result.json
      result.md
      imgs/                           (if layout enabled)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable

from glmocr import GlmOcr


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def sanitize_folder_name(name: str) -> str:
    name = name.strip()
    name = name.replace(os.sep, "_")
    name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", name)  # windows-illegal + control chars
    name = re.sub(r"\s+", " ", name)
    return name or "image"


def safe_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_images_in_dir(d: Path) -> list[Path]:
    files: list[Path] = []
    for p in d.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    return sorted(files)


def process_one_image(
    image_path: Path,
    out_root: Path,
    keep_images: bool,
    copy_image: bool,
    fail_fast: bool,
    ocr: GlmOcr,
) -> None:
    image_path = image_path.resolve()
    img_name = sanitize_folder_name(image_path.stem)
    img_out_dir = out_root / img_name
    img_out_dir.mkdir(parents=True, exist_ok=True)

    # Optionally copy original image into output folder
    if copy_image:
        dst = img_out_dir / image_path.name
        if dst.resolve() != image_path:
            shutil.copy2(image_path, dst)

    try:
        print(f"\n[INFO] Processing image: {image_path.name}")
        print(f"[INFO] Output folder: {img_out_dir}")

        print("[INFO] GLM-OCR parse...")
        result = ocr.parse(str(image_path))

        # Saves: result.json, result.md, imgs/ (if layout enabled)
        result.save(output_dir=str(img_out_dir))

        # Save a normalized version of the markdown (#7)
        try:
            from glmocr.postprocess.text_normalizer import normalize_text
            if result.markdown_result and result.markdown_result.strip():
                normalized_md = normalize_text(result.markdown_result)
                norm_path = img_out_dir / (sanitize_folder_name(image_path.stem) + ".normalized.md")
                norm_path.write_text(normalized_md, encoding="utf-8")
        except Exception:
            pass  # normalization is optional for image mode

        # Optional extra dump (ignore if SDK doesn't support)
        try:
            safe_write_json(img_out_dir / "result.full.to_dict.json", result.to_dict())
        except Exception:
            pass

        # If user wants outputs only (json/md) and not the original image copy,
        # remove copied image only when copy_image=False and keep_images=False does nothing.
        # keep_images controls *keeping original image in output* only if copy_image=True.
        if copy_image and (not keep_images):
            try:
                (img_out_dir / image_path.name).unlink(missing_ok=True)
            except Exception:
                pass

        print(f"[INFO] done ✅ -> {img_out_dir}")

    except Exception as ex:
        print(f"[ERROR] Image failed: {image_path}: {ex}", file=sys.stderr)
        if fail_fast:
            raise


def main() -> int:
    ap = argparse.ArgumentParser(description="Run GLM-OCR on images (file or folder).")

    inp = ap.add_mutually_exclusive_group(required=True)
    inp.add_argument("--image", help="Path to a single image")
    inp.add_argument("--image-dir", help="Path to a directory containing images")

    ap.add_argument("--out", default="./outputs_image", help="Output ROOT folder (default: ./outputs_image)")
    ap.add_argument("--keep-images", action="store_true", help="Keep copied original images inside each image folder")
    ap.add_argument("--copy-image", action="store_true", help="Copy input image into its output folder")
    ap.add_argument("--fail-fast", action="store_true", help="Stop on first failure")

    # GLM-OCR runtime
    ap.add_argument("--mode", choices=["maas", "selfhosted"], default=None, help="GLM-OCR mode")
    ap.add_argument("--api-key", default=None, help="MaaS API key (if mode=maas)")
    ap.add_argument("--config", default=None, help="Path to config.yaml (optional)")
    ap.add_argument("--enable-layout", action="store_true", help="Enable layout detection (if available)")
    ap.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING/ERROR")

    # Self-hosted OCR API location (vLLM/SGLang)
    ap.add_argument("--ocr-host", default=None, help="Selfhosted OCR API host (e.g., localhost)")
    ap.add_argument("--ocr-port", type=int, default=None, help="Selfhosted OCR API port (e.g., 8080)")

    args = ap.parse_args()

    out_root = Path(args.out).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Env overrides (SDK reads GLMOCR_*).
    if args.mode:
        os.environ["GLMOCR_MODE"] = args.mode
    if args.api_key:
        os.environ["GLMOCR_API_KEY"] = args.api_key
    if args.log_level:
        os.environ["GLMOCR_LOG_LEVEL"] = args.log_level
    if args.enable_layout:
        os.environ["GLMOCR_ENABLE_LAYOUT"] = "true"
    if args.ocr_host:
        os.environ["GLMOCR_OCR_API_HOST"] = args.ocr_host
    if args.ocr_port is not None:
        os.environ["GLMOCR_OCR_API_PORT"] = str(args.ocr_port)

    # GlmOcr init kwargs (optional; env vars also work)
    parser_kwargs = {}
    if args.config:
        parser_kwargs["config_path"] = str(Path(args.config).expanduser().resolve())
    if args.api_key:
        parser_kwargs["api_key"] = args.api_key
    if args.mode:
        parser_kwargs["mode"] = args.mode
    if args.enable_layout:
        parser_kwargs["enable_layout"] = True
    if args.log_level:
        parser_kwargs["log_level"] = args.log_level

    # Collect images
    images: list[Path] = []
    if args.image:
        p = Path(args.image).expanduser()
        if not p.exists() or not p.is_file():
            print(f"[ERROR] Image not found: {p}", file=sys.stderr)
            return 2
        if p.suffix.lower() not in IMAGE_EXTS:
            print(f"[ERROR] Unsupported image extension: {p.suffix}", file=sys.stderr)
            return 2
        images = [p]
    else:
        d = Path(args.image_dir).expanduser()
        if not d.exists() or not d.is_dir():
            print(f"[ERROR] Not a directory: {d}", file=sys.stderr)
            return 2
        images = iter_images_in_dir(d)
        if not images:
            print(f"[ERROR] No images found in: {d}", file=sys.stderr)
            print(f"[INFO] Supported: {sorted(IMAGE_EXTS)}")
            return 2

    print(f"[INFO] Output ROOT: {out_root}")
    print(f"[INFO] Images to process: {len(images)}")

    try:
        with GlmOcr(**parser_kwargs) as ocr:
            for img in images:
                try:
                    process_one_image(
                        image_path=img,
                        out_root=out_root,
                        keep_images=args.keep_images,
                        copy_image=args.copy_image,
                        fail_fast=args.fail_fast,
                        ocr=ocr,
                    )
                except Exception as e:
                    print(f"[ERROR] Image failed: {img.name}: {e}", file=sys.stderr)
                    if args.fail_fast:
                        return 1
                    continue

        print("\n[INFO] All done ✅")
        return 0

    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())