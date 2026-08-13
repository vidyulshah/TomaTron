import torch
import cv2
from PIL import Image
from pathlib import Path
from transformers import DetrImageProcessor, DetrForObjectDetection

script_dir = Path(__file__).parent

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_DIR       = script_dir / "tomato_detection_model"
VIDEO_PATH      = script_dir / "Classification_Images" / "test_video.mp4"   # ← update this
OUTPUT_PATH     = script_dir / "Classification_Images" / "result_video.mp4"
CONFIDENCE      = 0.5       # lower if too few detections, raise to reduce false positives
SAVE_OUTPUT     = True      # set False to only display without saving

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Colour per class in BGR (OpenCV format)
CLASS_COLORS = {
    "ripe"    : (0,   200,   0),   # green
    "semiripe": (0,   165, 255),   # orange
    "unripe"  : (0,     0, 255),   # red
}
DEFAULT_COLOR = (255, 255, 0)


def draw_detections(frame, results, id2label):
    """Draw boxes and labels on a frame, return annotated frame and per-class counts."""
    img    = frame.copy()
    h, w   = img.shape[:2]
    counts = {}

    for score, label_idx, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_name = id2label[label_idx.item()]
        confidence = score.item()
        color      = CLASS_COLORS.get(label_name, DEFAULT_COLOR)
        counts[label_name] = counts.get(label_name, 0) + 1

        x0, y0, x1, y1 = box.int().tolist()
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)

        cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness=2)

        text        = f"{label_name} {confidence:.0%}"
        font        = cv2.FONT_HERSHEY_SIMPLEX
        font_scale  = 0.55
        thickness   = 2
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

        cv2.rectangle(img, (x0, y0 - th - 8), (x0 + tw + 4, y0), color, -1)
        cv2.putText(img, text, (x0 + 2, y0 - 4), font, font_scale, (0, 0, 0), thickness)

    return img, counts


def draw_overlay(frame, counts, frame_num, fps):
    """Draw per-class count summary and frame info in the top-right corner."""
    h, w    = frame.shape[:2]
    lines   = [f"Frame: {frame_num}  FPS: {fps:.1f}"]
    total   = sum(counts.values())
    lines  += [f"Total : {total}"]
    lines  += [f"{label:<10}: {count}" for label, count in sorted(counts.items())]

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness  = 1
    padding    = 6
    line_h     = 22

    max_w = max(cv2.getTextSize(l, font, font_scale, thickness)[0][0] for l in lines)
    box_w = max_w + padding * 2
    box_h = line_h * len(lines) + padding

    x0 = w - box_w - 10
    y0 = 10
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (200, 200, 200), 1)

    for i, line in enumerate(lines):
        color = (0, 255, 255) if i == 0 else (255, 255, 255)
        cv2.putText(frame, line, (x0 + padding, y0 + padding + line_h * (i + 1) - 4),
                    font, font_scale, color, thickness)
    return frame


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

    # ── Open video ─────────────────────────────────────────────────────────────
    source = 0 if not VIDEO_PATH.exists() else str(VIDEO_PATH)
    if source == 0:
        print("Video file not found — opening webcam (press Q to quit).", flush=True)
    else:
        print(f"Processing video: {VIDEO_PATH.name}", flush=True)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    orig_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Resolution   : {orig_w} x {orig_h}", flush=True)
    print(f"Source FPS   : {src_fps:.1f}", flush=True)
    if total_frames > 0:
        print(f"Total Frames : {total_frames}", flush=True)

    # ── Video writer ───────────────────────────────────────────────────────────
    writer = None
    if SAVE_OUTPUT and source != 0:
        OUTPUT_PATH.parent.mkdir(exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, src_fps, (orig_w, orig_h))

    # ── Frame loop ─────────────────────────────────────────────────────────────
    import time
    frame_num  = 0
    fps_smooth = src_fps

    print("\nPress Q to quit ...\n", flush=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        frame_num += 1

        # Convert BGR → RGB PIL for processor
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        inputs = processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([pil_image.size[::-1]]).to(DEVICE)
        results      = processor.post_process_object_detection(
            outputs,
            target_sizes = target_sizes,
            threshold    = CONFIDENCE,
        )[0]

        annotated, counts = draw_detections(frame, results, id2label)
        elapsed           = time.perf_counter() - t0
        fps_smooth        = 0.9 * fps_smooth + 0.1 * (1.0 / max(elapsed, 1e-6))

        annotated = draw_overlay(annotated, counts, frame_num, fps_smooth)

        if writer:
            writer.write(annotated)

        # Scale display window to fit screen (cap at 1280px wide)
        disp_w = min(1280, orig_w)
        scale  = disp_w / orig_w
        disp   = cv2.resize(annotated, (disp_w, int(orig_h * scale)), interpolation=cv2.INTER_AREA)
        cv2.imshow("TomaTron — Video Detection (Q to quit)", disp)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Stopped by user.", flush=True)
            break

    # ── Cleanup ────────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
        print(f"\nSaved annotated video to: {OUTPUT_PATH}", flush=True)
    cv2.destroyAllWindows()
    print(f"Processed {frame_num} frames.", flush=True)


if __name__ == "__main__":
    main()