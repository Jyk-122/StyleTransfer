#!/usr/bin/env python3
r"""读取 DiffSynth-Studio metadata.json，横向拼接内容、风格、结果。

默认输出前 5 条记录，每条一张 PNG，同时生成一张总览图 overview.png。
图片按比例缩放、留白补齐，不裁剪；标注元数据索引、字段、路径及原始尺寸。
仅依赖 Pillow：pip install Pillow

示例：
    python preview_style_triplets.py \
        --metadata-path /data/StyleExpert-40K-diffsynth/metadata.json \
        --dataset-base-path /data/StyleExpert-40K-diffsynth \
        --output-dir /data/StyleExpert-40K-preview --count 5

增加 --random --seed 42 可随机抽样；--start-index 100 可从第 101 条开始。
不依赖 prepare 脚本或文件名规则，只读取每条记录的三个图片字段。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

from PIL import Image, ImageDraw, ImageFont, ImageOps


def load_font(size: int, custom: Path | None = None):
    if custom is not None:
        return ImageFont.truetype(str(custom), size)
    for name in ("DejaVuSans.ttf", "Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def wrap_label(text: str, font, width: int) -> list[str]:
    """Wrap long paths by measured width, without silently truncating filenames."""
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lines, line = [], ""
    for character in text:
        candidate = line + character
        bounds = draw.textbbox((0, 0), candidate, font=font)
        if line and bounds[2] - bounds[0] > width:
            lines.append(line)
            line = character
        else:
            line = candidate
    lines.append(line)
    return lines


def load_triplet(row: dict, index: int, base_path: Path):
    if not isinstance(row, dict):
        raise ValueError(f"metadata[{index}] 必须是对象")
    references = row.get("edit_image")
    if not isinstance(references, list) or len(references) != 2:
        raise ValueError(f"metadata[{index}].edit_image 必须是 [内容图, 风格图] 两个路径")
    names = [references[0], references[1], row.get("image")]
    images, resolved = [], []
    for role, name in zip(("content", "style", "target"), names):
        if not isinstance(name, str) or not name:
            raise ValueError(f"metadata[{index}] 的 {role} 路径无效：{name!r}")
        path = Path(name).expanduser()
        if not path.is_absolute():
            path = base_path / path
        path = path.resolve()
        try:
            with Image.open(path) as source:
                source.load()
                images.append(source.convert("RGB"))
        except Exception as exc:
            raise ValueError(f"metadata[{index}] 的 {role} 图片读取失败：{path}\n{exc}") from exc
        resolved.append(str(path))
    return names, images, resolved


def make_strip(index: int, names: list[str], images: list, cell_size: int, title_font, label_font):
    margin, gap = 16, 16
    title_height, role_height, line_height = 40, 38, 23
    labels = [wrap_label(name, label_font, cell_size - 16) for name in names]
    label_height = (max(map(len, labels)) + 1) * line_height + 16
    width = 2 * margin + 3 * cell_size + 2 * gap
    height = margin + title_height + role_height + cell_size + label_height + margin
    canvas = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), f"metadata[{index}]  |  Content -> Style reference -> Target", fill="#17202a", font=title_font)
    role_y = margin + title_height
    image_y = role_y + role_height
    titles = ("CONTENT  |  edit_image[0]", "STYLE  |  edit_image[1]", "TARGET  |  image")
    colors = ("#e5effc", "#efe7fa", "#e5f3e9")
    for column, (image, title, color, lines) in enumerate(zip(images, titles, colors, labels)):
        x = margin + column * (cell_size + gap)
        draw.rectangle((x, role_y, x + cell_size - 1, image_y - 1), fill=color)
        draw.text((x + 8, role_y + 7), title, fill="#17202a", font=label_font)
        draw.rectangle((x, image_y, x + cell_size - 1, image_y + cell_size - 1), fill="#f0f0f0")
        resized = ImageOps.contain(image, (cell_size, cell_size), method=Image.Resampling.LANCZOS)
        canvas.paste(resized, (x + (cell_size - resized.width) // 2, image_y + (cell_size - resized.height) // 2))
        y = image_y + cell_size + 8
        draw.text((x + 8, y), f"Original: {image.width} x {image.height}", fill="#58616b", font=label_font)
        for line in lines:
            y += line_height
            draw.text((x + 8, y), line, fill="#17202a", font=label_font)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--dataset-base-path", type=Path, help="应与 train.sh 的 dataset_base_path 一致；默认 metadata 所在目录")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--random", action="store_true", help="固定随机种子抽取样本")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cell-size", type=int, default=512, help="每张图的预览区域边长")
    parser.add_argument("--font", type=Path, help="自定义字体，中文路径可指定支持中文的字体")
    args = parser.parse_args()
    if args.count < 1 or args.start_index < 0 or args.cell_size < 384:
        parser.error("count 必须大于 0，start-index 不能为负，cell-size 至少为 384")
    if args.random and args.start_index:
        parser.error("--random 与非零 --start-index 不能同时使用")
    metadata_path = args.metadata_path.expanduser().resolve()
    base_path = (args.dataset_base_path or metadata_path.parent).expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    try:
        with metadata_path.open(encoding="utf-8-sig") as stream:
            metadata = json.load(stream)
        if not isinstance(metadata, list) or not metadata:
            raise ValueError("metadata.json 必须是非空数组")
        if args.start_index >= len(metadata):
            raise ValueError(f"start-index 超出记录数 {len(metadata)}")
        if args.random:
            indices = sorted(random.Random(args.seed).sample(range(len(metadata)), min(args.count, len(metadata))))
        else:
            indices = list(range(args.start_index, min(args.start_index + args.count, len(metadata))))
        title_font, label_font = load_font(22, args.font), load_font(18, args.font)
        strips, manifest = [], []
        for index in indices:
            names, images, resolved = load_triplet(metadata[index], index, base_path)
            strips.append(make_strip(index, names, images, args.cell_size, title_font, label_font))
            manifest.append({
                "metadata_index": index,
                "content": resolved[0], "style": resolved[1], "target": resolved[2],
                "preview": f"triplet_{index:06d}.png",
            })
        # Finish reading all requested samples before publishing any previews.
        output_dir.mkdir(parents=True, exist_ok=True)
        for strip, sample in zip(strips, manifest):
            path = output_dir / sample["preview"]
            strip.save(path)
            print(path)
        overview = Image.new("RGB", (max(s.width for s in strips), sum(s.height for s in strips) + 16 * (len(strips) - 1)), "#dfe3e8")
        y = 0
        for strip in strips:
            overview.paste(strip, (0, y))
            y += strip.height + 16
        overview.save(output_dir / "overview.png")
        with (output_dir / "preview_manifest.json").open("w", encoding="utf-8") as stream:
            json.dump({"metadata_path": str(metadata_path), "dataset_base_path": str(base_path), "samples": manifest}, stream, ensure_ascii=False, indent=2)
        print(f"已输出 {len(indices)} 组三元组；总览：{output_dir / 'overview.png'}")
        return 0
    except (OSError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
