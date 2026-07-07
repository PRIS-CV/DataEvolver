from __future__ import annotations

import hashlib
import shutil
import struct
import zlib
from pathlib import Path
from typing import Iterable, List

from ..prompt_writer import build_prompts
from ..schemas import DatasetRequest, DatasetSample, relpath
from .base import GeneratorAdapter


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def write_placeholder_png(path: Path, *, seed_text: str, width: int = 320, height: int = 240) -> None:
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    c1 = tuple(digest[:3])
    c2 = tuple(digest[3:6])
    c3 = tuple(digest[6:9])
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            if x < width // 3:
                color = c1
            elif x < 2 * width // 3:
                color = c2
            else:
                color = c3
            shade = 24 if ((x // 16) + (y // 16)) % 2 else 0
            row.extend(max(0, min(255, channel + shade)) for channel in color)
        rows.append(bytes(row))

    raw = b"".join(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        b"\x89PNG\r\n\x1a\n",
        _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
        _png_chunk(b"tEXt", b"Software\x00DataEvolver multimodal dry-run"),
        _png_chunk(b"IDAT", zlib.compress(raw, 9)),
        _png_chunk(b"IEND", b""),
    ]
    path.write_bytes(b"".join(payload))


def write_placeholder_gif(path: Path, *, seed_text: str, width: int = 320, height: int = 240, frames: int = 8) -> None:
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    path.parent.mkdir(parents=True, exist_ok=True)
    width = max(64, min(65535, int(width)))
    height = max(64, min(65535, int(height)))
    color_a = digest[:3]
    color_b = digest[3:6]
    header = b"GIF89a" + struct.pack("<HH", width, height) + b"\x80\x00\x00" + color_a + color_b
    comment = (
        b"DataEvolver multimodal dry-run placeholder video; "
        + f"frames={max(2, int(frames))}; ".encode("ascii")
    )
    blocks = []
    while len(b"".join(blocks)) < 1536:
        chunk = (comment + digest.hex().encode("ascii"))[:255]
        blocks.append(b"\x21\xFE" + bytes([len(chunk)]) + chunk + b"\x00")
    path.write_bytes(header + b"".join(blocks) + b"\x3B")


def iter_images(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


class DryRunGenerator(GeneratorAdapter):
    name = "dryrun"

    def generate(self, request: DatasetRequest, output_root: Path) -> List[DatasetSample]:
        if request.route == "t2i":
            return self._generate_t2i(request, output_root)
        if request.route == "edit":
            return self._generate_edit(request, output_root)
        if request.route == "blender":
            return self._generate_blender_stub(request, output_root)
        if request.route == "t2v":
            return self._generate_t2v_stub(request, output_root)
        raise ValueError(f"Unsupported route for dry-run: {request.route}")

    def _base_review(self) -> dict:
        return {
            "status": "pending",
            "enabled": False,
            "reason": "VLM evaluation is reserved for a later phase",
        }

    def _generate_t2i(self, request: DatasetRequest, output_root: Path) -> List[DatasetSample]:
        samples = []
        for prompt_row in build_prompts(request, request.num_samples):
            sample_id = prompt_row["sample_id"]
            out_path = output_root / "samples" / "images" / f"{sample_id}.png"
            existed = out_path.exists()
            if not (request.skip_existing and existed):
                write_placeholder_png(
                    out_path,
                    seed_text=f"{request.seed}:{sample_id}:{prompt_row['prompt']}",
                    width=request.width,
                    height=request.height,
                )
            samples.append(
                DatasetSample(
                    sample_id=sample_id,
                    route=request.route,
                    prompt=prompt_row["prompt"],
                    output_path=relpath(out_path, output_root),
                    generator=self.name,
                    status="skipped_existing" if existed and request.skip_existing else "generated",
                    metadata={
                        "dry_run": True,
                        "backend": "placeholder_png",
                        "intended_model": request.model_name,
                        "intended_model_path": request.model_path,
                        "original_prompt": prompt_row.get("original_prompt", prompt_row["prompt"]),
                        "rewritten_prompt": prompt_row.get("rewritten_prompt", prompt_row["prompt"]),
                        "constraints": prompt_row.get("constraints", []),
                        "tier": prompt_row.get("tier"),
                        "source_prompt_id": prompt_row.get("source_prompt_id"),
                        "rewrite_reason": prompt_row.get("rewrite_reason"),
                    },
                    vlm_review=self._base_review(),
                )
            )
        return samples

    def _generate_edit(self, request: DatasetRequest, output_root: Path) -> List[DatasetSample]:
        input_dir = Path(request.input_image_dir or "")
        if not input_dir.exists() or not input_dir.is_dir():
            raise FileNotFoundError(f"input image directory not found: {input_dir}")

        image_paths = list(iter_images(input_dir))[: request.num_samples]
        if not image_paths:
            raise FileNotFoundError(f"no images found in input image directory: {input_dir}")

        prompt_rows = build_prompts(request, len(image_paths))
        samples = []
        for prompt_row, src in zip(prompt_rows, image_paths):
            sample_id = prompt_row["sample_id"]
            out_path = output_root / "samples" / "edited" / f"{sample_id}{src.suffix.lower()}"
            existed = out_path.exists()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not (request.skip_existing and existed):
                shutil.copy2(src, out_path)
            samples.append(
                DatasetSample(
                    sample_id=sample_id,
                    route=request.route,
                    prompt=prompt_row["prompt"],
                    input_path=str(src),
                    output_path=relpath(out_path, output_root),
                    edit_instruction=prompt_row["edit_instruction"],
                    generator=self.name,
                    status="skipped_existing" if existed and request.skip_existing else "generated",
                    metadata={
                        "dry_run": True,
                        "backend": "copy_input_image",
                        "intended_model": request.model_name,
                        "intended_model_path": request.model_path,
                        "source_filename": src.name,
                        "original_edit_instruction": prompt_row.get(
                            "original_edit_instruction",
                            prompt_row["edit_instruction"],
                        ),
                        "model_edit_instruction": prompt_row["edit_instruction"],
                        "constraints": prompt_row.get("constraints", []),
                        "tier": prompt_row.get("tier"),
                        "source_prompt_id": prompt_row.get("source_prompt_id"),
                        "rewrite_reason": prompt_row.get("rewrite_reason"),
                    },
                    vlm_review=self._base_review(),
                )
            )
        return samples

    def _generate_blender_stub(self, request: DatasetRequest, output_root: Path) -> List[DatasetSample]:
        prompt_row = build_prompts(request, 1)[0]
        marker = output_root / "samples" / "blender" / f"{prompt_row['sample_id']}.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            "{\n"
            '  "status": "dryrun_stub",\n'
            '  "intended_route": "existing Blender pipeline",\n'
            '  "real_entrypoints": ["src/dataevolver/cli/run_all.sh"]\n'
            "}\n",
            encoding="utf-8",
        )
        return [
            DatasetSample(
                sample_id=prompt_row["sample_id"],
                route=request.route,
                prompt=prompt_row["prompt"],
                output_path=relpath(marker, output_root),
                generator=self.name,
                status="stub",
                metadata={
                    "dry_run": True,
                    "backend": "blender_route_marker",
                    "intended_model": request.model_name,
                    "intended_model_path": request.model_path,
                },
                vlm_review=self._base_review(),
            )
        ]

    def _generate_t2v_stub(self, request: DatasetRequest, output_root: Path) -> List[DatasetSample]:
        samples = []
        for prompt_row in build_prompts(request, request.num_samples):
            sample_id = prompt_row["sample_id"]
            out_path = output_root / "samples" / "videos" / f"{sample_id}.gif"
            existed = out_path.exists()
            if not (request.skip_existing and existed):
                write_placeholder_gif(
                    out_path,
                    seed_text=f"{request.seed}:{sample_id}:{prompt_row['prompt']}",
                    width=request.width,
                    height=request.height,
                    frames=min(max(request.num_frames, 2), 16),
                )
            samples.append(
                DatasetSample(
                    sample_id=sample_id,
                    route=request.route,
                    prompt=prompt_row["prompt"],
                    output_path=relpath(out_path, output_root),
                    generator=self.name,
                    status="skipped_existing" if existed and request.skip_existing else "generated",
                    metadata={
                        "dry_run": True,
                        "backend": "placeholder_gif_video",
                        "intended_model": request.model_name,
                        "intended_model_path": request.model_path,
                        "original_prompt": prompt_row.get("original_prompt", prompt_row["prompt"]),
                        "rewritten_prompt": prompt_row.get("rewritten_prompt", prompt_row["prompt"]),
                        "negative_prompt": prompt_row.get("negative_prompt", request.negative_prompt),
                        "constraints": prompt_row.get("constraints", []),
                        "num_frames": request.num_frames,
                        "fps": request.fps,
                        "tier": prompt_row.get("tier"),
                        "source_prompt_id": prompt_row.get("source_prompt_id"),
                        "rewrite_reason": prompt_row.get("rewrite_reason"),
                    },
                    vlm_review=self._base_review(),
                )
            )
        return samples
