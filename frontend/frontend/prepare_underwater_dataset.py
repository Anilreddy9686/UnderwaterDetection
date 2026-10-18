from __future__ import annotations

import argparse
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

import yaml

BASE_DIR = Path(__file__).resolve().parent
URPC_DIR = BASE_DIR.parent.parent / "backend" / "URPC2020-2"
OUTPUT_DIR = BASE_DIR / "training_data" / "expanded"

CLASS_NAMES = [
    "echinus",
    "holothurian",
    "scallop",
    "starfish",
    "plastic",
    "marine_life",
    "rov",
    "other_debris",
]

LABEL_MAP = {
    "plastic": "plastic",
    "platstic": "plastic",  # normalize source typo
    "bio": "marine_life",
    "rov": "rov",
    "cloth": "other_debris",
    "fishing": "other_debris",
    "metal": "other_debris",
    "paper": "other_debris",
    "papper": "other_debris",  # normalize source typo
    "rubber": "other_debris",
    "wood": "other_debris",
    "unknown": "other_debris",
}


def image_entries(archive: zipfile.ZipFile):
    for info in archive.infolist():
        parts = Path(info.filename).parts
        if len(parts) >= 2 and parts[0].lower() in {"train", "val", "test"}:
            if Path(info.filename).suffix.lower() in {".jpg", ".jpeg", ".png"}:
                yield info, parts[0].lower()


def convert_annotation(xml_bytes: bytes, counters: Counter[str]) -> list[str]:
    root = ET.fromstring(xml_bytes)
    size = root.find("size")
    if size is None:
        return []
    width = float(size.findtext("width", "0"))
    height = float(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        return []

    rows: list[str] = []
    class_ids = {name: index for index, name in enumerate(CLASS_NAMES)}
    for obj in root.findall("object"):
        source_label = (obj.findtext("name") or "").strip().lower()
        target_label = LABEL_MAP.get(source_label)
        if target_label is None:
            continue  # e.g. timestamp overlays are not detector targets
        box = obj.find("bndbox")
        if box is None:
            continue
        try:
            x1 = float(box.findtext("xmin", "0"))
            y1 = float(box.findtext("ymin", "0"))
            x2 = float(box.findtext("xmax", "0"))
            y2 = float(box.findtext("ymax", "0"))
        except ValueError:
            continue
        x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
        y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            continue
        cx = ((x1 + x2) / 2.0) / width
        cy = ((y1 + y2) / 2.0) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        rows.append(f"{class_ids[target_label]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        counters[target_label] += 1
    return rows


def prepare(archive_path: Path, output_dir: Path = OUTPUT_DIR) -> Path:
    if not archive_path.is_file():
        raise FileNotFoundError(f"Dataset ZIP not found: {archive_path}")
    if not URPC_DIR.is_dir():
        raise FileNotFoundError(f"Original URPC dataset not found: {URPC_DIR}")

    trash_dir = output_dir / "trash"
    if trash_dir.exists():
        shutil.rmtree(trash_dir)
    for split in ("train", "val", "test"):
        (trash_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (trash_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    missing_xml = 0
    with zipfile.ZipFile(archive_path) as archive:
        entries = list(image_entries(archive))
        if not entries:
            raise ValueError("No train/val/test images found in the ZIP.")
        for info, split in entries:
            source_path = Path(info.filename)
            xml_name = str(source_path.with_suffix(".xml")).replace("\\", "/")
            try:
                xml_bytes = archive.read(xml_name)
            except KeyError:
                rows = []
                missing_xml += 1
            else:
                rows = convert_annotation(xml_bytes, counts)

            ext = source_path.suffix.lower()
            target_stem = f"trash_{split}_{source_path.stem}"
            target_image = trash_dir / split / "images" / f"{target_stem}{ext}"
            target_label = trash_dir / split / "labels" / f"{target_stem}.txt"
            target_image.write_bytes(archive.read(info))
            target_label.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    manifests = {}
    for split, original_split in (("train", "train"), ("val", "valid"), ("test", "test")):
        original_images = URPC_DIR / original_split / "images"
        original_labels = URPC_DIR / original_split / "labels"
        if not original_images.is_dir() or not original_labels.is_dir():
            raise FileNotFoundError(f"Original URPC {original_split} images/labels are missing.")
        images = sorted(p.resolve() for p in original_images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        images += sorted(p.resolve() for p in (trash_dir / split / "images").iterdir() if p.is_file())
        manifest = output_dir / f"{split}.txt"
        manifest.write_text("\n".join(p.as_posix() for p in images) + "\n", encoding="utf-8")
        manifests[split] = manifest.name

    config = {
        "path": output_dir.resolve().as_posix(),
        "train": manifests["train"],
        "val": manifests["val"],
        "test": manifests["test"],
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    report = [
        "Expanded dataset prepared.",
        f"Config: {yaml_path}",
        f"Image files in archive: {len(entries)}",
        f"Missing XML annotations: {missing_xml}",
        "Combined classes: " + ", ".join(CLASS_NAMES),
        "Source annotation counts included:",
    ]
    report.extend(f"  {name}: {counts[name]}" for name in CLASS_NAMES[4:])
    report.append("Rock class: not included; the downloaded archive has no labeled rock boxes.")
    report_text = "\n".join(report) + "\n"
    (output_dir / "PREPARATION_REPORT.txt").write_text(report_text, encoding="utf-8")
    print(report_text)
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge the downloaded underwater debris data with the existing URPC dataset.")
    parser.add_argument("--archive", type=Path, default=Path.home() / "Downloads" / "archive.zip", help="Path to the downloaded archive.zip")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="Output folder for the expanded training dataset")
    args = parser.parse_args()
    prepare(args.archive.expanduser().resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
