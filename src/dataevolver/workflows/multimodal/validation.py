from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import struct
import subprocess


def _luma(pixel: tuple[int, int, int]) -> float:
    r, g, b = pixel
    return 0.299 * r + 0.587 * g + 0.114 * b


def _foreground_stats(img) -> Dict[str, object]:
    rgb = img.convert("RGB")
    width, height = rgb.size
    sample_w = min(128, width)
    sample_h = min(128, height)
    small = rgb.resize((sample_w, sample_h))
    pixels = list(small.getdata())
    if not pixels:
        return {}

    border = []
    for y in range(sample_h):
        for x in range(sample_w):
            if x in {0, sample_w - 1} or y in {0, sample_h - 1}:
                border.append(pixels[y * sample_w + x])
    if not border:
        border = pixels
    bg = tuple(sum(p[i] for p in border) / len(border) for i in range(3))

    threshold_sq = 24.0 * 24.0
    fg_points = []
    for idx, pixel in enumerate(pixels):
        dist_sq = sum((pixel[i] - bg[i]) ** 2 for i in range(3))
        if dist_sq > threshold_sq:
            x = idx % sample_w
            y = idx // sample_w
            fg_points.append((x, y, pixel))

    fg_fraction = len(fg_points) / float(sample_w * sample_h)
    result: Dict[str, object] = {
        "estimated_background_rgb": [round(v, 3) for v in bg],
        "foreground_fraction": round(fg_fraction, 5),
    }
    if fg_points:
        xs = [p[0] for p in fg_points]
        ys = [p[1] for p in fg_points]
        cxs = [(p[0] + 0.5) / sample_w for p in fg_points]
        cys = [(p[1] + 0.5) / sample_h for p in fg_points]
        result.update(
            {
                "foreground_bbox_xyxy": [
                    round(min(xs) / sample_w, 4),
                    round(min(ys) / sample_h, 4),
                    round((max(xs) + 1) / sample_w, 4),
                    round((max(ys) + 1) / sample_h, 4),
                ],
                "foreground_center_xy": [
                    round(sum(cxs) / len(cxs), 4),
                    round(sum(cys) / len(cys), 4),
                ],
                "foreground_mean_rgb": [
                    round(sum(p[2][i] for p in fg_points) / len(fg_points), 3)
                    for i in range(3)
                ],
            }
        )
    return result


def validate_image(path: Optional[Path], *, expected_width: int | None = None, expected_height: int | None = None) -> Dict[str, object]:
    if path is None:
        return {"status": "fail", "issues": ["missing_path"]}
    issues = []
    path = Path(path)
    if not path.exists():
        return {"status": "fail", "issues": ["missing_file"], "path": str(path)}
    size = path.stat().st_size
    if size < 1024:
        issues.append("output_too_small")

    result: Dict[str, object] = {"path": str(path), "file_size": size}
    try:
        from PIL import Image, ImageStat

        with Image.open(path) as img:
            img.load()
            width, height = img.size
            result.update({"width": width, "height": height, "mode": img.mode})
            if expected_width and width != expected_width:
                issues.append("width_mismatch")
            if expected_height and height != expected_height:
                issues.append("height_mismatch")
            stat = ImageStat.Stat(img.convert("RGB"))
            means = [round(float(v), 3) for v in stat.mean]
            stddev = [round(float(v), 3) for v in stat.stddev]
            result.update({"mean_rgb": means, "stddev_rgb": stddev})
            luma_mean = round(_luma(tuple(means)), 3)
            result["luma_mean"] = luma_mean
            result.update(_foreground_stats(img))
            if max(stddev) < 1.0:
                issues.append("low_information_image")
            if max(means) < 3.0:
                issues.append("near_black")
            if min(means) > 252.0:
                issues.append("near_white")
            if luma_mean > 245.0 and max(stddev) < 18.0:
                issues.append("washed_out_low_contrast")
            foreground_fraction = float(result.get("foreground_fraction") or 0.0)
            if foreground_fraction < 0.005:
                issues.append("foreground_missing")
            elif foreground_fraction < 0.035:
                issues.append("foreground_too_small")
            if foreground_fraction < 0.08 and luma_mean > 235.0:
                issues.append("weak_subject_separation")
    except Exception as exc:
        parsed = False
        try:
            head = path.read_bytes()[:24]
            if head.startswith(b"\x89PNG\r\n\x1a\n") and head[12:16] == b"IHDR":
                width, height = struct.unpack(">II", head[16:24])
                result.update({"width": int(width), "height": int(height), "mode": "unknown_png"})
                if expected_width and width != expected_width:
                    issues.append("width_mismatch")
                if expected_height and height != expected_height:
                    issues.append("height_mismatch")
                parsed = True
        except Exception:
            parsed = False
        if not parsed:
            issues.append("image_open_failed")
            result["error"] = str(exc)

    result["issues"] = issues
    result["status"] = "pass" if not issues else ("warn" if issues == ["output_too_small"] else "fail")
    return result


def _probe_gif(path: Path) -> Dict[str, object]:
    from PIL import Image, ImageSequence

    with Image.open(path) as img:
        frames = [frame.copy().convert("RGB") for frame in ImageSequence.Iterator(img)]
        width, height = img.size
    result: Dict[str, object] = {
        "container": "gif",
        "width": width,
        "height": height,
        "num_frames": len(frames),
    }
    if frames:
        first = frames[0]
        result.update({"first_frame": validate_image(path, expected_width=width, expected_height=height)})
        if len(frames) > 1:
            result["has_multiple_frames"] = True
    return result


def _probe_with_imageio(path: Path) -> Dict[str, object]:
    import imageio.v3 as iio

    props = iio.improps(path)
    meta = iio.immeta(path, exclude_applied=False)
    shape = getattr(props, "shape", None)
    result: Dict[str, object] = {"container": path.suffix.lower().lstrip(".") or "video"}
    if shape and len(shape) >= 3:
        result["num_frames"] = int(shape[0])
        result["height"] = int(shape[1])
        result["width"] = int(shape[2])
    if meta:
        result["metadata"] = {str(k): str(v) for k, v in list(meta.items())[:20]}
    return result


def _probe_with_ffprobe(path: Path) -> Dict[str, object]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,r_frame_rate,duration",
        "-of",
        "default=noprint_wrappers=1",
        str(path),
    ]
    result = subprocess.run(cmd, text=True, capture_output=True, timeout=20)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffprobe failed")
    parsed: Dict[str, object] = {"container": path.suffix.lower().lstrip(".") or "video"}
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in {"width", "height", "nb_frames"} and value.isdigit():
            parsed["num_frames" if key == "nb_frames" else key] = int(value)
        else:
            parsed[key] = value
    return parsed


def validate_video(
    path: Optional[Path],
    *,
    expected_width: int | None = None,
    expected_height: int | None = None,
    expected_num_frames: int | None = None,
) -> Dict[str, object]:
    if path is None:
        return {"status": "fail", "issues": ["missing_path"]}
    issues = []
    path = Path(path)
    if not path.exists():
        return {"status": "fail", "issues": ["missing_file"], "path": str(path)}
    size = path.stat().st_size
    result: Dict[str, object] = {"path": str(path), "file_size": size}
    small_file_threshold = 1024 if path.suffix.lower() == ".gif" else 4096
    if size < small_file_threshold:
        issues.append("output_too_small")
    suffix = path.suffix.lower()
    if suffix not in {".mp4", ".gif", ".webm", ".mov"}:
        issues.append("unexpected_video_extension")

    probe_errors = []
    for probe in (
        (_probe_gif if suffix == ".gif" else None),
        _probe_with_imageio,
        _probe_with_ffprobe,
    ):
        if probe is None:
            continue
        try:
            result.update(probe(path))
            break
        except Exception as exc:
            probe_errors.append(f"{getattr(probe, '__name__', 'probe')}: {exc}")
    if "width" not in result or "height" not in result:
        issues.append("video_metadata_unavailable")
        result["probe_errors"] = probe_errors[-3:]

    width = result.get("width")
    height = result.get("height")
    frames = result.get("num_frames")
    if expected_width and isinstance(width, int) and width != expected_width:
        issues.append("width_mismatch")
    if expected_height and isinstance(height, int) and height != expected_height:
        issues.append("height_mismatch")
    if expected_num_frames and isinstance(frames, int) and frames < max(2, min(expected_num_frames, 8)):
        issues.append("too_few_frames")

    result["issues"] = issues
    metadata_only_issues = {"video_metadata_unavailable"}
    result["status"] = "pass" if not issues or set(issues) <= metadata_only_issues else "fail"
    return result
