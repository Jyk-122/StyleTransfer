#!/usr/bin/env python3
r"""将官方 StyleExpert-40K 转换为 DiffSynth-Studio 的多图编辑训练数据。

依赖：pip install Pillow tqdm

直接读取 Hugging Face 下载目录（包含 metadata.jsonl 和 image_shards/）：
    python prepare_styleexpert40k.py \
        --dataset-dir /data/datasets/StyleExpert \
        --output-dir /data/datasets/StyleExpert-40K-diffsynth --workers 16

已经解压图片时，增加 --images-dir /data/style_expert40k_images。
可用 --limit 32 先转换前 32 条记录；每次请使用新的输出目录。

官方字段映射：content_path -> edit_image[0]，style_path -> edit_image[1]，
target -> image。图片角色由 metadata.jsonl 决定，不根据文件名猜测。
官方 prompt 是内容描述；默认改用与原训练一致的风格迁移指令。
--include-content-caption 可将原内容描述追加到指令末尾。

分卷 style_expert40k.tar.gz.00、.01……是同一个 gzip 包的连续字节切片。
本脚本串流拼接并解压，不创建额外的约 98 GB 合并压缩包。
也支持已经合并的 style_expert40k.tar.gz。

输出：content/、style/、stylized/、metadata.json、conversion_report.json。
所有唯一图片都执行完整解码；任意一张坏图会导致相关三元组被剔除。
缺图默认报错，--allow-missing 可明确允许剔除缺图三元组。
默认优先硬链接，跨文件系统时自动复制；--copy 创建独立副本。
原始数据不被修改，已有 metadata.json 不被覆盖。此脚本不下载数据。

数据来源：https://huggingface.co/datasets/HH-LG/StyleExpert
"""

from __future__ import annotations

import argparse
import errno
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile
from concurrent.futures import ProcessPoolExecutor

from PIL import Image
from tqdm import tqdm


DEFAULT_PROMPT = (
    "Transfer the visual style of image 2 to image 1 "
    "while preserving the content, structure, objects and layout of image 1."
)
FIELD_FOLDERS = {
    "content_path": "content",
    "style_path": "style",
    "target": "stylized",
}
ARCHIVE_NAME = "style_expert40k.tar.gz"
BUFFER_SIZE = 1024 * 1024


def relative_name(value: str) -> str:
    """Normalize ./name without allowing absolute paths or directory traversal."""
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise ValueError(f"无效的图片相对路径：{value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"图片路径必须是安全的相对路径：{value!r}")
    return path.as_posix()


def load_records(metadata_path: Path, limit: int | None = None) -> list[dict]:
    records = []
    with metadata_path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("记录必须是 JSON 对象")
                for field in FIELD_FOLDERS:
                    row[field] = relative_name(row[field])
                if not isinstance(row.get("prompt", ""), str):
                    raise ValueError("prompt 必须是字符串")
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{metadata_path} 第 {line_number} 行：{exc}") from exc
            row["_source_line"] = line_number
            records.append(row)
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"元数据没有可用记录：{metadata_path}")
    return records


def build_destinations(records: list[dict], output_dir: Path) -> dict[str, list[Path]]:
    destinations: dict[str, list[Path]] = {}
    for row in records:
        for field, folder in FIELD_FOLDERS.items():
            name = row[field]
            path = output_dir / folder / name
            paths = destinations.setdefault(name, [])
            if path not in paths:
                paths.append(path)
    return destinations


def find_archive_parts(dataset_dir: Path) -> list[Path]:
    for directory in (dataset_dir / "image_shards", dataset_dir):
        parts = [
            path for path in directory.glob(ARCHIVE_NAME + ".*")
            if path.is_file() and path.name[len(ARCHIVE_NAME) + 1:].isdigit()
        ]
        if not parts:
            continue
        parts.sort(key=lambda path: int(path.name.rsplit(".", 1)[1]))
        indices = [int(path.name.rsplit(".", 1)[1]) for path in parts]
        if indices != list(range(len(parts))):
            raise ValueError(f"压缩分卷编号不连续，必须从 .00 开始：{indices}")
        with parts[0].open("rb") as stream:
            if stream.read(2) != b"\x1f\x8b":
                raise ValueError(f"首个分卷不是 gzip 数据（请检查下载是否完整）：{parts[0]}")
        return parts
    for directory in (dataset_dir, dataset_dir / "image_shards"):
        archive = directory / ARCHIVE_NAME
        if archive.is_file():
            return [archive]
    raise FileNotFoundError(
        f"在 {dataset_dir} 中没有找到 {ARCHIVE_NAME} 或连续分卷。"
        "若已手动解压，请通过 --images-dir 指定图片根目录。"
    )


class JoinedParts(io.RawIOBase):
    """Expose split archive files as one bounded-memory, forward-only stream."""

    def __init__(self, parts: list[Path]):
        super().__init__()
        self.parts = iter(parts)
        self.current = None

    def readable(self):
        return True

    def readinto(self, buffer):
        view = memoryview(buffer)
        count = 0
        while count < len(view):
            if self.current is None:
                path = next(self.parts, None)
                if path is None:
                    break
                self.current = path.open("rb")
            size = self.current.readinto(view[count:])
            if size:
                count += size
            else:
                self.current.close()
                self.current = None
        return count

    def close(self):
        if self.current is not None:
            self.current.close()
            self.current = None
        super().close()


def write_member(archive: tarfile.TarFile, member: tarfile.TarInfo, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        # Never extract paths, symlinks or permissions from the archive itself.
        with archive.extractfile(member) as source:
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as target:
                temporary = Path(target.name)
                shutil.copyfileobj(source, target, length=BUFFER_SIZE)
                if target.tell() != member.size:
                    raise OSError(f"图片数据不完整：{member.name}")
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def extract_needed(parts: list[Path], destinations: dict[str, list[Path]]) -> dict[str, Path]:
    remaining = set(destinations)
    found = {}
    print(f"串流读取 {len(parts)} 个压缩文件，共 {sum(p.stat().st_size for p in parts) / 1024**3:.2f} GiB")
    with JoinedParts(parts) as raw, io.BufferedReader(raw, BUFFER_SIZE) as stream:
        with tarfile.open(fileobj=stream, mode="r|gz", bufsize=BUFFER_SIZE) as archive:
            with tqdm(total=len(remaining), desc="Extracting unique images", unit="img") as progress:
                for member in archive:
                    if not member.isfile():
                        continue
                    name = relative_name(member.name)
                    if name not in remaining:
                        continue
                    destination = destinations[name][0]
                    write_member(archive, member, destination)
                    found[name] = destination
                    remaining.remove(name)
                    progress.update()
                    if not remaining:
                        # A --limit subset need not decompress the rest of the dataset.
                        break
    return found


def check_image(item: tuple[str, Path]) -> tuple[str, str | None]:
    name, path = item
    try:
        with Image.open(path) as image:
            image.load()  # Full decode, including truncated/corrupt pixel data.
            if min(image.size) < 16:
                raise ValueError(f"图片尺寸 {image.size} 小于训练器需要的最小边长 16")
        return name, None
    except Exception as exc:
        return name, f"{type(exc).__name__}: {exc}"


def validate_images(sources: dict[str, Path], workers: int) -> dict[str, str]:
    bad = {}
    items = sorted(sources.items())
    if workers == 1:
        results = map(check_image, items)
        for name, error in tqdm(results, total=len(items), desc="Checking unique images", unit="img"):
            if error is not None:
                bad[name] = error
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = executor.map(check_image, items, chunksize=32)
            for name, error in tqdm(results, total=len(items), desc="Checking unique images", unit="img"):
                if error is not None:
                    bad[name] = error
    return bad


def materialize(source: Path, destination: Path, copy: bool):
    if destination.exists() and os.path.samefile(source, destination):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(dir=destination.parent)
    os.close(fd)
    temporary = Path(filename)
    try:
        if copy:
            shutil.copyfile(source, temporary)
        else:
            temporary.unlink()
            try:
                os.link(source, temporary)
            except OSError as exc:
                if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP, errno.EMLINK):
                    raise
                shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, payload):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def convert(args) -> dict:
    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    metadata_path = (args.metadata_path or dataset_dir / "metadata.jsonl").expanduser().resolve()
    metadata_output = output_dir / "metadata.json"
    if metadata_output.exists():
        raise FileExistsError(f"输出已经存在：{metadata_output}；请选择新的 --output-dir。")
    records = load_records(metadata_path, args.limit)
    destinations = build_destinations(records, output_dir)
    if args.images_dir is not None:
        images_dir = args.images_dir.expanduser().resolve()
        if not images_dir.is_dir():
            raise NotADirectoryError(f"图片根目录不存在：{images_dir}")
        # Avoid replacing original images when the user chooses an overlapping output.
        if images_dir == output_dir or output_dir in images_dir.parents or images_dir in output_dir.parents:
            raise ValueError("--images-dir 与 --output-dir 不能相同，也不能互相包含。")
        parts = None
    else:
        parts = find_archive_parts(dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"记录数：{len(records)}；需要检查的唯一图片数：{len(destinations)}")
    report = {
        "source_metadata": str(metadata_path),
        "output_dir": str(output_dir),
        "status": "incomplete",
        "selected_triplets": len(records),
        "unique_referenced_images": len(destinations),
    }
    report_path = output_dir / "conversion_report.json"
    try:
        if parts is not None:
            sources = extract_needed(parts, destinations)
        else:
            sources = {name: images_dir / name for name in destinations if (images_dir / name).is_file()}
        missing = sorted(set(destinations) - set(sources))
        corrupt = validate_images(sources, args.workers)
        invalid = set(missing) | set(corrupt)
        kept = [row for row in records if not any(row[field] in invalid for field in FIELD_FOLDERS)]
        report.update({
            "found_unique_images": len(sources),
            "valid_triplets": len(kept),
            "skipped_triplets": len(records) - len(kept),
            "missing_images": missing,
            "corrupt_images": corrupt,
            "skipped_source_lines": [
                row["_source_line"] for row in records
                if any(row[field] in invalid for field in FIELD_FOLDERS)
            ],
        })
        if missing and not args.allow_missing:
            raise ValueError(
                f"缺少 {len(missing)} 张引用图片，未生成训练 metadata.json。"
                "请检查解压目录或分卷是否完整；若确实只处理部分图片，可加 --allow-missing。"
            )
        if not kept:
            raise ValueError("没有完整、可解码的三元组，未生成训练 metadata.json。")
        needed = build_destinations(kept, output_dir)
        for name, paths in tqdm(needed.items(), desc="Organizing images", unit="img"):
            for destination in paths:
                materialize(sources[name], destination, args.copy)
        metadata = []
        for row in kept:
            prompt = args.prompt
            if args.include_content_caption and row.get("prompt", "").strip():
                prompt += " Content of image 1: " + row["prompt"].strip()
            metadata.append({
                "image": (PurePosixPath("stylized") / row["target"]).as_posix(),
                "prompt": prompt,
                "edit_image": [
                    (PurePosixPath("content") / row["content_path"]).as_posix(),
                    (PurePosixPath("style") / row["style_path"]).as_posix(),
                ],
            })
        write_json(metadata_output, metadata)
        report["status"] = "complete"
        write_json(report_path, report)
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        write_json(report_path, report)
        raise
    print(f"完成：保留 {len(kept)} 组，剔除 {len(records) - len(kept)} 组；缺图 {len(missing)} 张，坏图 {len(corrupt)} 张")
    print(f"训练元数据：{metadata_output}")
    print(f"检查报告：{report_path}")
    print("训练参数：")
    print(f"  --dataset_base_path {output_dir}")
    print(f"  --dataset_metadata_path {metadata_output}")
    print('  --data_file_keys "image,edit_image"')
    print('  --extra_inputs "edit_image"')
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="官方数据下载目录", default="./datasets/StyleExpert")
    parser.add_argument("--output-dir", type=Path, required=True, help="整理后的数据目录", default="./datasets/StyleExpert-40k")
    parser.add_argument("--metadata-path", type=Path, help="默认使用 dataset-dir/metadata.jsonl")
    parser.add_argument("--images-dir", type=Path, help="已解压图片根目录；设置后不读取压缩包")
    parser.add_argument("--workers", type=int, default=16, help="完整解码检查的进程数，默认 16")
    parser.add_argument("--limit", type=int, help="只处理前 N 条记录，用于小规模验证")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="训练用的风格迁移指令")
    parser.add_argument("--include-content-caption", action="store_true", help="在指令后追加官方内容描述")
    parser.add_argument("--copy", action="store_true", help="复制图片，避免共享硬链接")
    parser.add_argument("--allow-missing", action="store_true", help="允许剔除缺图三元组并输出其余数据")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--workers 和 --limit 必须是正整数")
    if not args.prompt.strip():
        parser.error("--prompt 不能为空")
    try:
        convert(args)
    except (OSError, ValueError, tarfile.TarError, EOFError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
