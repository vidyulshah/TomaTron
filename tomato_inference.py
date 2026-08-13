import torch
import cv2
import json
from PIL import Image
from pathlib import Path
from transformers import DetrImageProcessor, DetrForObjectDetection

script_dir = Path(__file__).parent

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_DIR        = script_dir / "tomato_detection_model"
TEST_IMAGE_PATH  = script_dir / "Classification_Images" / "test_tomato.png"
OUTPUT_IMAGE_PATH= script_dir / "Classification_Images" / "result.jpg"
CONFIDENCE       = 0.5      # raise to 0.7 to reduce false positives

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Colour per class in BGR (OpenCV format)
CLASS_COLORS = {
    "ripe"    : (0,   200,   0),   # green
    "semiripe": (0,   165, 255),   # orange
    "unripe"  : (0,     0, 255),   # red
}
DEFAULT_COLOR = (255, 255, 0)      # cyan fallback for unknown labels


def draw_detections(cv_img, results, id2label):
    """Draw bounding boxes and labels on a copy of cv_img. Returns annotated image."""
    img       = cv_img.copy()
    h, w      = img.shape[:2]
    counts    = {}

    for score, label_idx, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_name  = id2label[label_idx.item()]
        confidence  = score.item()
        color       = CLASS_COLORS.get(label_name, DEFAULT_COLOR)
        counts[label_name] = counts.get(label_name, 0) + 1

        # box is [x_min, y_min, x_max, y_max] in absolute pixels
        x0, y0, x1, y1 = box.int().tolist()
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)

        # Bounding box
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness=2)

        # Label text with shadow for readability
        text        = f"{label_name} {confidence:.0%}"
        font        = cv2.FONT_HERSHEY_SIMPLEX
        font_scale  = 0.6
        thickness   = 2
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

        # Filled background behind text
        cv2.rectangle(img, (x0, y0 - th - 8), (x0 + tw + 4, y0), color, -1)
        cv2.putText(img, text, (x0 + 2, y0 - 4), font, font_scale, (0, 0, 0), thickness)

    return img, counts


def print_summary(counts, total):
    print("\n" + "=" * 45)
    print(f"  TOTAL TOMATOES DETECTED : {total}")
    print("-" * 45)
    for label, count in sorted(counts.items()):
        bar = "█" * count
        print(f"  {label:<12} {count:>3}  {bar}")
    print("=" * 45 + "\n")


def main():
    # ── Load model ─────────────────────────────────────────────────────────────
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model not found at: {MODEL_DIR}\nRun tomato_training.py first.")

    print(f"Using Device : {DEVICE}", flush=True)
    print(f"Loading model from {MODEL_DIR} ...", flush=True)

    processor = DetrImageProcessor.from_pretrained(MODEL_DIR)
    model     = DetrForObjectDetection.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()

    id2label = model.config.id2label

    # ── Load image ─────────────────────────────────────────────────────────────
    if not TEST_IMAGE_PATH.exists():
        raise FileNotFoundError(f"Test image not found at: {TEST_IMAGE_PATH}")

    print(f"Running inference on: {TEST_IMAGE_PATH.name}", flush=True)

    pil_image = Image.open(TEST_IMAGE_PATH).convert("RGB")
    cv_image  = cv2.imread(str(TEST_IMAGE_PATH))

    # ── Inference ──────────────────────────────────────────────────────────────
    inputs = processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Convert outputs → boxes in pixel coords, filtered by confidence
    target_sizes = torch.tensor([pil_image.size[::-1]]).to(DEVICE)   # [H, W]
    results      = processor.post_process_object_detection(
        outputs,
        target_sizes = target_sizes,
        threshold    = CONFIDENCE,
    )[0]

    total = len(results["scores"])

    # ── Display results ────────────────────────────────────────────────────────
    if total == 0:
        print(f"\nNo tomatoes detected above {CONFIDENCE:.0%} confidence.")
        print("Try lowering the CONFIDENCE threshold at the top of the script.")
    else:
        annotated, counts = draw_detections(cv_image, results, id2label)
        print_summary(counts, total)

        # Resize for display (keep aspect ratio, cap at 900px tall)
        disp_h    = min(900, annotated.shape[0])
        scale     = disp_h / annotated.shape[0]
        disp_w    = int(annotated.shape[1] * scale)
        display   = cv2.resize(annotated, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

        cv2.imshow("TomaTron — DETR Detection", display)
        print("Press any key to close the window ...", flush=True)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save result
        OUTPUT_IMAGE_PATH.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(OUTPUT_IMAGE_PATH), annotated)
        print(f"Result saved to: {OUTPUT_IMAGE_PATH}", flush=True)


if __name__ == "__main__":
    main()