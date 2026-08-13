"""
Convert VIA (VGG Image Annotator) JSON exports to COCO detection format.

Usage:
    python via_to_coco.py

Reads:
    dataset/train/train_json.json
    dataset/val/val_json.json

Writes:
    dataset/train/train_coco.json
    dataset/val/val_coco.json

Polygon regions are converted to their enclosing axis-aligned bounding box,
which is what DETR needs for object detection. The original polygon is kept
in the "segmentation" field in case you want instance segmentation later.
"""

import json
from pathlib import Path
from PIL import Image

script_dir = Path(__file__).parent

SPLITS = [
    (script_dir / "dataset" / "train", "train_json.json", "train_coco.json"),
    (script_dir / "dataset" / "val",   "val_json.json",   "val_coco.json"),
]

CLASS_ATTRIBUTE = "names"   # the key inside region_attributes holding the label


def collect_class_names(split_paths):
    """Scan every split first so train and val share identical category IDs."""
    names = set()
    for image_dir, via_name, _ in split_paths:
        via_path = image_dir / via_name
        if not via_path.exists():
            continue
        with open(via_path) as f:
            via = json.load(f)
        for entry in via.values():
            for region in entry.get("regions", []):
                label = region.get("region_attributes", {}).get(CLASS_ATTRIBUTE)
                if label:
                    names.add(label.strip())
    return sorted(names)


def polygon_to_bbox(shape):
    """Return COCO [x, y, w, h] from any VIA shape type."""
    kind = shape.get("name")

    if kind == "polygon" or kind == "polyline":
        xs, ys = shape["all_points_x"], shape["all_points_y"]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    elif kind == "rect":
        x0, y0 = shape["x"], shape["y"]
        x1, y1 = x0 + shape["width"], y0 + shape["height"]
    elif kind == "circle":
        r = shape["r"]
        x0, y0 = shape["cx"] - r, shape["cy"] - r
        x1, y1 = shape["cx"] + r, shape["cy"] + r
    elif kind == "ellipse":
        rx, ry = shape["rx"], shape["ry"]
        x0, y0 = shape["cx"] - rx, shape["cy"] - ry
        x1, y1 = shape["cx"] + rx, shape["cy"] + ry
    else:
        return None

    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def polygon_to_segmentation(shape):
    """Flat [x1, y1, x2, y2, ...] list, or None for non-polygon shapes."""
    if shape.get("name") in ("polygon", "polyline"):
        xs, ys = shape["all_points_x"], shape["all_points_y"]
        flat = []
        for x, y in zip(xs, ys):
            flat.extend([float(x), float(y)])
        return [flat]
    return None


def convert(image_dir, via_name, out_name, categories, cat_name_to_id):
    via_path = image_dir / via_name
    with open(via_path) as f:
        via = json.load(f)

    images, annotations = [], []
    ann_id = 1
    skipped_missing = []
    skipped_regions = 0

    for image_id, (key, entry) in enumerate(sorted(via.items()), start=1):
        filename = entry["filename"]
        img_path = image_dir / filename

        if not img_path.exists():
            skipped_missing.append(filename)
            continue

        with Image.open(img_path) as im:
            width, height = im.size

        regions = entry.get("regions", [])

        # VIA sometimes stores regions as a dict keyed by index rather than a list
        if isinstance(regions, dict):
            regions = list(regions.values())

        kept = 0
        for region in regions:
            shape = region.get("shape_attributes", {})
            label = region.get("region_attributes", {}).get(CLASS_ATTRIBUTE)

            if not label:
                skipped_regions += 1
                continue

            bbox = polygon_to_bbox(shape)
            if bbox is None or bbox[2] <= 0 or bbox[3] <= 0:
                skipped_regions += 1
                continue

            # Clamp to image bounds — hand-drawn polygons often spill over the edge
            x, y, w, h = bbox
            x = max(0.0, min(x, width))
            y = max(0.0, min(y, height))
            w = min(w, width - x)
            h = min(h, height - y)
            if w <= 0 or h <= 0:
                skipped_regions += 1
                continue

            ann = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": cat_name_to_id[label.strip()],
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
            }

            seg = polygon_to_segmentation(shape)
            if seg:
                ann["segmentation"] = seg

            annotations.append(ann)
            ann_id += 1
            kept += 1

        images.append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height,
        })

    coco = {
        "info": {"description": f"Converted from VIA export {via_name}"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    out_path = image_dir / out_name
    with open(out_path, "w") as f:
        json.dump(coco, f, indent=2)

    images_with_anns = len({a["image_id"] for a in annotations})

    print(f"\n{via_name} -> {out_name}")
    print(f"  images written      : {len(images)}")
    print(f"  images with boxes   : {images_with_anns}")
    print(f"  annotations         : {len(annotations)}")
    if skipped_regions:
        print(f"  regions skipped     : {skipped_regions} (no label or bad shape)")
    if skipped_missing:
        preview = ", ".join(skipped_missing[:5])
        more = f" (+{len(skipped_missing) - 5} more)" if len(skipped_missing) > 5 else ""
        print(f"  MISSING image files : {len(skipped_missing)} -> {preview}{more}")

    return out_path


def main():
    class_names = collect_class_names(SPLITS)

    if not class_names:
        raise SystemExit(
            f"No labels found under region_attributes['{CLASS_ATTRIBUTE}']. "
            "Open the VIA JSON and check the attribute name."
        )

    # COCO category IDs are conventionally 1-indexed; the training script
    # remaps them to 0-indexed for DETR on its own.
    categories = [
        {"id": i, "name": name, "supercategory": "tomato"}
        for i, name in enumerate(class_names, start=1)
    ]
    cat_name_to_id = {c["name"]: c["id"] for c in categories}

    print("Classes found:")
    for c in categories:
        print(f"  {c['id']}: {c['name']}")

    for image_dir, via_name, out_name in SPLITS:
        if not (image_dir / via_name).exists():
            print(f"\nSkipping {image_dir / via_name} — file not found")
            continue
        convert(image_dir, via_name, out_name, categories, cat_name_to_id)

    print("\nDone. Point TRAIN_JSON / VAL_JSON at the *_coco.json files.")


if __name__ == "__main__":
    main()