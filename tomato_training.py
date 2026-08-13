import os, json, torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.optim as optim
from transformers import DetrImageProcessor, DetrForObjectDetection
from tqdm import tqdm
from pathlib import Path

script_dir = Path(__file__).parent

# ── Paths ──────────────────────────────────────────────────────────────────────
TRAIN_IMAGE_DIR = script_dir / "dataset" / "train"
VAL_IMAGE_DIR   = script_dir / "dataset" / "val"
TRAIN_JSON      = script_dir / "dataset" / "train" / "train_coco.json"   # from via_to_coco.py
VAL_JSON        = script_dir / "dataset" / "val"   / "val_coco.json"     # from via_to_coco.py
OUTPUT_DIR      = script_dir / "tomato_detection_model"

# ── Hyperparameters ────────────────────────────────────────────────────────────
BATCH_SIZE    = 2        # 8 GB VRAM on the 4060; raise to 4 only if nothing else uses the GPU
EPOCHS        = 10
LEARNING_RATE = 1e-4
MAX_GRAD_NORM = 0.1      # DETR's reference implementation clips here; prevents early divergence

# Cap input resolution to fit in VRAM. DETR's default is 800/1333, which will OOM
# on an 8 GB card at batch size 2. Set to None to use the processor's defaults.
IMAGE_SIZE = {"shortest_edge": 480, "longest_edge": 800}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using Device: {DEVICE}", flush=True)
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)


# ── Dataset ────────────────────────────────────────────────────────────────────
class TomatoDataset(Dataset):
    def __init__(self, image_dir, json_path, processor):
        self.image_dir = Path(image_dir)
        self.processor = processor

        with open(json_path) as f:
            coco = json.load(f)

        # Map COCO category IDs → 0-indexed labels (DETR expects 0-indexed)
        cat_ids_sorted       = sorted(cat["id"] for cat in coco["categories"])
        self.cat_id_to_label = {cat_id: idx for idx, cat_id in enumerate(cat_ids_sorted)}
        self.id2label        = {
            idx: next(c["name"] for c in coco["categories"] if c["id"] == cat_id)
            for cat_id, idx in self.cat_id_to_label.items()
        }
        self.label2id = {v: k for k, v in self.id2label.items()}

        # Image id → image info
        self.images = {img["id"]: img for img in coco["images"]}

        # Image id → list of annotations
        self.annotations = {}
        for ann in coco["annotations"]:
            self.annotations.setdefault(ann["image_id"], []).append(ann)

        # Only keep images that have at least one annotation
        self.image_ids = [img_id for img_id in self.images if img_id in self.annotations]

        print(f"Loaded {len(self.image_ids)} images", flush=True)
        print(f"Classes: {self.id2label}", flush=True)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id   = self.image_ids[idx]
        img_info = self.images[img_id]
        image    = Image.open(self.image_dir / img_info["file_name"]).convert("RGB")
        anns     = self.annotations[img_id]

        targets = {
            "image_id"   : img_id,
            "annotations": [
                {
                    "bbox"       : ann["bbox"],                               # [x, y, w, h] COCO format
                    "category_id": self.cat_id_to_label[ann["category_id"]], # remapped to 0-indexed
                    "area"       : ann["area"],
                    "iscrowd"    : ann.get("iscrowd", 0),
                }
                for ann in anns
            ],
        }

        encoding = self.processor(images=image, annotations=targets, return_tensors="pt")
        # Remove the batch dimension added by the processor
        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels"      : encoding["labels"][0],
        }


def collate_fn(batch):
    """
    Custom collate function — images are padded to a common size and stacked,
    but labels stay as a list because each image has a different number of boxes.
    """
    pixel_values = [b["pixel_values"] for b in batch]

    # Images in a batch can differ in size after resizing, so pad to the largest.
    max_h = max(p.shape[1] for p in pixel_values)
    max_w = max(p.shape[2] for p in pixel_values)

    padded, masks = [], []
    for p in pixel_values:
        c, h, w = p.shape
        canvas = torch.zeros((c, max_h, max_w), dtype=p.dtype)
        canvas[:, :h, :w] = p
        padded.append(canvas)

        mask = torch.zeros((max_h, max_w), dtype=torch.long)
        mask[:h, :w] = 1
        masks.append(mask)

    return {
        "pixel_values": torch.stack(padded),
        "pixel_mask"  : torch.stack(masks),
        "labels"      : [b["labels"] for b in batch],
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    model_name = "facebook/detr-resnet-50"

    if IMAGE_SIZE is not None:
        processor = DetrImageProcessor.from_pretrained(model_name, size=IMAGE_SIZE)
    else:
        processor = DetrImageProcessor.from_pretrained(model_name)

    print("\nLoading datasets ...", flush=True)
    train_dataset = TomatoDataset(TRAIN_IMAGE_DIR, TRAIN_JSON, processor)
    val_dataset   = TomatoDataset(VAL_IMAGE_DIR,   VAL_JSON,   processor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    num_labels = len(train_dataset.id2label)
    print(f"\nNumber of classes: {num_labels}", flush=True)

    model = DetrForObjectDetection.from_pretrained(
        model_name,
        num_labels             = num_labels,
        id2label               = train_dataset.id2label,
        label2id               = train_dataset.label2id,
        ignore_mismatched_sizes= True,
    )
    model.to(DEVICE)

    # Use separate learning rates:
    # - Backbone (pretrained ResNet) gets a 10x smaller LR to preserve learned features
    # - Transformer head gets the full LR to learn the new detection task
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if "backbone" in n],     "lr": LEARNING_RATE * 0.1},
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n], "lr": LEARNING_RATE},
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)

    print("\nStarting Training .....", flush=True)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):

        # ── Training loop ──────────────────────────────────────────────────────
        model.train()
        train_loss = 0
        train_bar  = tqdm(train_loader, desc=f"{epoch+1}/{EPOCHS} [TRAIN]", leave=True)

        for batch in train_bar:
            pixel_values = batch["pixel_values"].to(DEVICE)
            pixel_mask   = batch["pixel_mask"].to(DEVICE)
            labels       = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]]

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

            # DETR computes loss internally (box L1 + GIoU + classification)
            loss = outputs.loss
            loss.backward()

            # Clip gradients — DETR is prone to loss spikes without this
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)

            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # ── Validation loop ────────────────────────────────────────────────────
        model.eval()
        val_loss = 0
        val_bar  = tqdm(val_loader, desc=f"{epoch+1}/{EPOCHS} [VAL]", leave=True)

        with torch.no_grad():
            for batch in val_bar:
                pixel_values = batch["pixel_values"].to(DEVICE)
                pixel_mask   = batch["pixel_mask"].to(DEVICE)
                labels       = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]]
                outputs      = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
                val_loss    += outputs.loss.item()
                val_bar.set_postfix(loss=f"{outputs.loss.item():.4f}")

        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch+1}/{EPOCHS} → "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}",
            flush=True
        )

        # ── Save best checkpoint ───────────────────────────────────────────────
        # Saving the *best* model rather than the last one matters on a small
        # dataset, where later epochs are often worse than earlier ones.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            OUTPUT_DIR.mkdir(exist_ok=True)
            model.save_pretrained(OUTPUT_DIR)
            processor.save_pretrained(OUTPUT_DIR)
            print(f"  ✓ saved — best val loss so far ({best_val_loss:.4f})", flush=True)

    print(f"\nTraining Complete! Best val loss: {best_val_loss:.4f}", flush=True)
    print(f"Model saved to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()