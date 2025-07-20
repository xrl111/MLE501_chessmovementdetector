import os
import logging
import shutil
import cv2
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import random
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_dataset(dataset_path):
    """Inspect class IDs from detection labels and verify dataset balance."""
    class_ids = defaultdict(int)
    sample_labels = []
    for split in ['train', 'valid']:
        label_dir = os.path.join(dataset_path, split, 'labels')
        if not os.path.exists(label_dir):
            logger.warning(f"Label directory not found: {label_dir}")
            continue
        for label_file in os.listdir(label_dir):
            if not label_file.endswith('.txt'):
                continue
            with open(os.path.join(label_dir, label_file), 'r') as f:
                lines = f.readlines()
                if not lines:
                    continue
                try:
                    class_id = int(lines[0].strip().split()[0])
                    class_ids[class_id] += 1
                    if len(sample_labels) < 5:
                        sample_labels.append(f"{split}/{label_file}: {lines[0].strip()}")
                except (ValueError, IndexError):
                    logger.warning(f"Invalid label format in {label_file}")
    logger.info(f"Class ID distribution: {dict(class_ids)}")
    logger.info("Sample labels:\n" + "\n".join(sample_labels))

    if class_ids:
        counts = np.array(list(class_ids.values()))
        imbalance_ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
        if imbalance_ratio > 5:
            logger.warning(f"Class imbalance detected (ratio: {imbalance_ratio:.2f}). Consider balancing dataset.")
    return class_ids

def preprocess_dataset(dataset_path, output_path, class_names, custom_dataset_path="/content/drive/MyDrive/MSE24_HN/chess_dataset"):
    """Convert detection dataset to classification dataset, limit original images to 50 per class, incorporate custom files, and limit final output to 50 images per class."""
    # Label remapping for original dataset
    label_remap = {
        "B": "b",       # black bishop
        "K": "board",   # board (ignored)
        "N": "k",       # black king
        "P": "n",       # black knight
        "Q": "p",       # black pawn
        "R": "q",       # black queen
        "b": "B",       # white bishop
        "board": "r",   # black rook
        "k": "K",       # white king
        "n": "N",       # white knight
        "p": "P",       # white pawn
        "q": "Q",       # white queen
        "r": "R"        # white rook
    }
    excluded_class = "board"
    remapped_class_names = sorted({v for k, v in label_remap.items() if v != excluded_class})

    # File-to-folder mapping for custom files
    file_mapping = {
        'bb_g.png': 'b', 'K.png': 'K', 'nb_g.png': 'n', 'N.png': 'N', 'pb_g.png': 'p',
        'rb.png': 'r', 'B_g.png': 'B', 'rb_g.png': 'r', 'kb.png': 'k', 'qb.png': 'q',
        'N_g.png': 'N', 'Q_g.png': 'Q', 'pb.png': 'p', 'B.png': 'B', 'qb_g.png': 'q',
        'bb.png': 'b', 'nb.png': 'n', 'R.png': 'R', 'R_g.png': 'R', 'kb_g.png': 'k',
        'P_g.png': 'P', 'K_g.png': 'K', 'P.png': 'P', 'Q.png': 'Q'
    }

    # Initialize image counts and temporary storage
    image_counts = {'train': defaultdict(int), 'valid': defaultdict(int)}
    temp_image_paths = {'train': defaultdict(list), 'valid': defaultdict(list)}
    original_class_counts = defaultdict(int)  # Track original images per class

    # Create output directories
    for split in ['train', 'valid']:
        for cls in remapped_class_names:
            os.makedirs(os.path.join(output_path, split, cls), exist_ok=True)

    # Process original dataset (limit to 50 images per class)
    ORIGINAL_LIMIT_PER_CLASS = 50
    for split in ['train', 'valid']:
        image_dir = os.path.join(dataset_path, split, 'images')
        label_dir = os.path.join(dataset_path, split, 'labels')

        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            logger.warning(f"Missing image or label directory for {split}")
            continue

        for img_file in os.listdir(image_dir):
            if not img_file.lower().endswith(('.jpg', '.png')):
                continue

            img_path = os.path.join(image_dir, img_file)
            label_path = os.path.join(label_dir, img_file.rsplit('.', 1)[0] + '.txt')

            if not os.path.exists(label_path):
                logger.warning(f"Missing label file for {img_file}")
                continue

            with open(label_path, 'r') as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) != 5:
                    logger.warning(f"Invalid label format in {label_path}, line {i+1}")
                    continue

                try:
                    class_id, x_center, y_center, width, height = map(float, parts)
                    class_id = int(class_id)
                    if class_id >= len(class_names):
                        logger.warning(f"Class ID {class_id} exceeds class_names length in {label_path}")
                        continue

                    orig_class = class_names[class_id]
                    mapped_class = label_remap.get(orig_class, orig_class)

                    if mapped_class == excluded_class:
                        continue

                    # Skip if class limit reached
                    if original_class_counts[mapped_class] >= ORIGINAL_LIMIT_PER_CLASS:
                        logger.debug(f"Skipping {img_file} for {mapped_class}: class limit ({ORIGINAL_LIMIT_PER_CLASS}) reached")
                        continue

                    img = cv2.imread(img_path)
                    if img is None:
                        logger.warning(f"Failed to load image {img_path}")
                        continue

                    h, w = img.shape[:2]
                    x_center *= w
                    y_center *= h
                    width *= w
                    height *= h
                    x1 = int(max(x_center - width / 2, 0))
                    y1 = int(max(y_center - height / 2, 0))
                    x2 = int(min(x_center + width / 2, w))
                    y2 = int(min(y_center + height / 2, h))

                    cropped = img[y1:y2, x1:x2]
                    if cropped.size == 0:
                        logger.warning(f"Empty crop for {img_file}, skipping")
                        continue

                    # Convert to grayscale and resize
                    cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                    cropped = cv2.cvtColor(cropped_gray, cv2.COLOR_GRAY2BGR)
                    cropped = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_AREA)

                    save_name = f"orig_{img_file.rsplit('.', 1)[0]}_{i}.jpg"
                    save_path = os.path.join(output_path, split, mapped_class, save_name)
                    temp_image_paths[split][mapped_class].append((cropped, save_path))
                    image_counts[split][mapped_class] += 1
                    original_class_counts[mapped_class] += 1
                    logger.debug(f"Prepared original image {save_name} for {mapped_class} in {split} (Total for {mapped_class}: {original_class_counts[mapped_class]})")

                except Exception as e:
                    logger.error(f"Error processing {img_file}, line {i+1}: {e}")
                    continue

    logger.info(f"Original images processed per class: {dict(original_class_counts)}")

    # Process custom files
    custom_files_processed = 0
    if os.path.exists(custom_dataset_path):
        logger.info(f"Processing custom files from {custom_dataset_path}")
        for file_name in os.listdir(custom_dataset_path):
            if file_name not in file_mapping:
                logger.warning(f"File {file_name} not in file_mapping, skipping")
                continue

            target_class = file_mapping[file_name]
            if target_class not in remapped_class_names:
                logger.warning(f"Target class {target_class} for {file_name} not in remapped class names, skipping")
                continue

            src_path = os.path.join(custom_dataset_path, file_name)
            img = cv2.imread(src_path)
            if img is None:
                logger.warning(f"Failed to load custom image {src_path}")
                continue

            # Convert to grayscale and resize
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

            save_name = f"custom_{file_name.rsplit('.', 1)[0]}.jpg"
            split = 'train' if random.random() < 0.8 else 'valid'
            save_path = os.path.join(output_path, split, target_class, save_name)
            temp_image_paths[split][target_class].append((img, save_path))
            image_counts[split][target_class] += 1
            custom_files_processed += 1
            logger.info(f"Added custom file {file_name} to {split}/{target_class} as {save_name}")
    else:
        logger.error(f"Custom dataset path {custom_dataset_path} does not exist")

    logger.info(f"Total custom files processed: {custom_files_processed}")

    # Limit to 50 images per class in final output
    for split in ['train', 'valid']:
        for cls in remapped_class_names:
            images = temp_image_paths[split][cls]
            logger.info(f"{split}/{cls}: {len(images)} images before final selection")
            if len(images) > 50:
                images = random.sample(images, 50)
                logger.info(f"Reduced {split}/{cls} to 50 images")
            for img, save_path in images:
                cv2.imwrite(save_path, img)
                logger.debug(f"Saved {os.path.basename(save_path)} to {save_path}")
            image_counts[split][cls] = len(images)

    logger.info(f"Final image counts per class: {dict(image_counts['train'])} for train, {dict(image_counts['valid'])} for valid")
    return image_counts

def main():
    # Step 1: Install dependencies
    os.system("pip install -q ultralytics opencv-python numpy torch")

    # Step 2: Download dataset
    try:
        os.system('curl -L "https://universe.roboflow.com/ds/lEIeDLYdtb?key=ytHQpJZNeT" -o roboflow.zip')
        os.system('unzip -q roboflow.zip')
        os.system('rm roboflow.zip')
        dataset_path = "/content"
        logger.info(f"Dataset downloaded to {dataset_path}")
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        return

    # Step 3: Validate structure
    if not os.path.exists(os.path.join(dataset_path, 'train')) or not os.path.exists(os.path.join(dataset_path, 'valid')):
        logger.error("Missing 'train' or 'valid' folders")
        return

    # Step 4: Define class names
    class_names = ['b', 'k', 'n', 'p', 'q', 'r', 'B', 'K', 'N', 'P', 'Q', 'R', 'board']

    # Step 5: Inspect dataset
    class_ids = inspect_dataset(dataset_path)
    if max(class_ids.keys(), default=-1) >= len(class_names):
        logger.warning("Class IDs exceed available class_names list. Check alignment.")

    # Step 6: Preprocess to classification dataset
    output_path = "/content/chess_classification_dataset"
    image_counts = preprocess_dataset(dataset_path, output_path, class_names)

    # Step 7: Final checks
    train_dir = os.path.join(output_path, 'train')
    valid_dir = os.path.join(output_path, 'valid')
    if not os.path.exists(train_dir) or not os.path.exists(valid_dir):
        logger.error("Preprocessed folders missing.")
        return
    if sum(image_counts['train'].values()) == 0 or sum(image_counts['valid'].values()) == 0:
        logger.error("No images after preprocessing. Aborting.")
        return

    # Step 8: Load classifier and train
    try:
        model = YOLO("yolov8n-cls.pt")
        logger.info("YOLOv8n-cls model loaded.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    try:
        # Check for GPU availability
        device = 0 if torch.cuda.is_available() else 'cpu'
        logger.info(f"Training on device: {device}")

        model.train(
            data=output_path,
            epochs=50,
            imgsz=224,
            batch=32,
            name="chess_piece_classifier",
            patience=10,
            device=device,
            optimizer="AdamW",
            lr0=0.001,
            cos_lr=True,
            augment=True,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            flipud=0.5,
            fliplr=0.5,
            mosaic=0.5,
            mixup=0.2
        )
        logger.info("Training completed.")

        # Step 9: Evaluate
        metrics = model.val()
        logger.info(f"Validation Top-1 Accuracy: {metrics.top1 * 100:.2f}%")
        logger.info(f"Validation Top-5 Accuracy: {metrics.top5 * 100:.2f}%")

        # Step 10: Test on sample images
        confidence_threshold = 0.3
        for split in ['train', 'valid']:
            for cls in os.listdir(os.path.join(output_path, split)):
                cls_dir = os.path.join(output_path, split, cls)
                if not os.path.isdir(cls_dir):
                    continue
                for img_file in os.listdir(cls_dir)[:2]:
                    img_path = os.path.join(cls_dir, img_file)
                    results = model(img_path)
                    confidence = results[0].probs.top1conf.item()
                    if confidence < confidence_threshold:
                        logger.warning(f"Low confidence ({confidence:.2f}) for {img_file}, skipping")
                        continue
                    pred_class = results[0].names[results[0].probs.top1]
                    logger.info(f"{img_file} → {pred_class} ({confidence:.2f})")

        # Step 11: Export
        model.export(format="onnx")
        logger.info("Model exported to ONNX.")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return

if __name__ == "__main__":
    main()
