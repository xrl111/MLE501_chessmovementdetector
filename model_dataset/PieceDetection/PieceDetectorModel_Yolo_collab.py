import os
import logging
import cv2
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import random
import torch
import gc
from albumentations import Compose, ColorJitter, GaussNoise, RandomBrightnessContrast
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Setup logging
logging.basicConfig(level=logging.INFO)  # Reduced verbosity
logger = logging.getLogger(__name__)

def inspect_dataset(dataset_path, class_names):
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
                    if len(sample_labels) < 3:  # Reduced sample size
                        sample_labels.append(f"{split}/{label_file}: {lines[0].strip()} (class {class_names[class_id]})")
                except (ValueError, IndexError):
                    logger.warning(f"Invalid label format in {label_file}")
    logger.info(f"Class ID distribution: {dict(class_ids)}")
    logger.info("Sample labels:\n" + "\n".join(sample_labels))
    return class_ids

def process_image(args):
    """Process a single image: crop, resize, augment, and save."""
    img_path, label_path, output_path, target_split, mapped_class, aug_count, augment = args
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None, None, None
        h, w = img.shape[:2]
        with open(label_path, 'r') as f:
            lines = f.readlines()
        results = []
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, x_center, y_center, width, height = map(float, parts)
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
                continue
            cropped = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_AREA)
            if mapped_class.islower():
                img_yuv = cv2.cvtColor(cropped, cv2.COLOR_BGR2YUV)
                img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
                cropped = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            save_name = f"orig_{os.path.basename(img_path).rsplit('.', 1)[0]}_{i}.jpg"
            save_path = os.path.join(output_path, target_split, mapped_class, save_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, cropped)
            results.append((target_split, mapped_class, 1))
            for aug_idx in range(aug_count):
                augmented = augment(image=cropped)
                img_aug = augmented['image']
                save_name_aug = f"orig_{os.path.basename(img_path).rsplit('.', 1)[0]}_{i}_aug{aug_idx}.jpg"
                save_path_aug = os.path.join(output_path, target_split, mapped_class, save_name_aug)
                cv2.imwrite(save_path_aug, img_aug)
                results.append((target_split, mapped_class, 1))
        return results, mapped_class, len(lines)
    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return None, None, None

def preprocess_dataset(dataset_path, output_path, class_names, custom_dataset_path="/content/drive/MyDrive/MSE24_HN/chess_dataset"):
    """Convert dataset to classification format, apply augmentation, and ensure class balance."""
    label_remap = {
        'B': 'b', 'K': 'board', 'N': 'k', 'P': 'n', 'Q': 'p', 'R': 'q',
        'b': 'B', 'k': 'K', 'n': 'N', 'p': 'P', 'q': 'Q', 'r': 'R',
        'board': 'r'
    }
    excluded_class = "excluded"
    remapped_class_names = sorted([c for c in class_names if c != 'board'])
    file_mapping = {
        'K.png': 'K', 'N.png': 'N', 'B_g.png': 'B', 'N_g.png': 'N', 'Q_g.png': 'Q',
        'B.png': 'B', 'R.png': 'R', 'R_g.png': 'R', 'P_g.png': 'P', 'K_g.png': 'K',
        'P.png': 'P', 'Q.png': 'Q', 'pb_g.png': 'p', 'rb.png': 'r', 'kb.png': 'k',
        'qb.png': 'q', 'qb_g.png': 'q', 'bb_g.png': 'b', 'nb_g.png': 'n', 'rb_g.png': 'r',
        'kb_g.png': 'k', 'pb.png': 'p', 'bb.png': 'b', 'nb.png': 'n',
        'custom_bb.png': 'b', 'b_extra.png': 'b', 'b_extra2.png': 'b', 'custom_bb_g.png': 'b',
        'custom_nb.png': 'n', 'n_new1.png': 'n', 'n_new2.png': 'n', 'custom_nb_g.png': 'n',
        'custom_kb.png': 'k', 'k_extra.png': 'k', 'k_extra2.png': 'k', 'custom_kb_g.png': 'k',
        'custom_pb.png': 'p', 'p_new1.png': 'p', 'p_new2.png': 'p', 'custom_pb_g.png': 'p',
        'custom_rb.png': 'r', 'r_extra.png': 'r', 'r_extra2.png': 'r', 'custom_rb_g.png': 'r',
        'custom_qb.png': 'q', 'q_extra.png': 'q', 'q_extra2.png': 'q', 'custom_qb_g.png': 'q'
    }

    augment = Compose([
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),  # Reduced intensity
        GaussNoise(p=0.3),
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)
    ])

    image_counts = {'train': defaultdict(int), 'valid': defaultdict(int)}
    original_class_counts = defaultdict(int)
    remapped_class_counts = defaultdict(int)
    custom_files_processed = defaultdict(list)
    r_files_found = []

    for split in ['train', 'valid']:
        for cls in remapped_class_names:
            os.makedirs(os.path.join(output_path, split, cls), exist_ok=True)

    board_class_count = 0
    if dataset_path and os.path.exists(dataset_path):
        for split in ['train', 'valid']:
            image_dir = os.path.join(dataset_path, split, 'images')
            label_dir = os.path.join(dataset_path, split, 'labels')
            if not os.path.exists(image_dir) or not os.path.exists(label_dir):
                logger.warning(f"Missing image or label directory for {split}")
                continue
            image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))]
            tasks = []
            for img_file in image_files:
                img_path = os.path.join(image_dir, img_file)
                label_path = os.path.join(label_dir, img_file.rsplit('.', 1)[0] + '.txt')
                if not os.path.exists(label_path):
                    continue
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                if not lines:
                    continue
                class_id = int(lines[0].strip().split()[0])
                if class_id >= len(class_names):
                    logger.warning(f"Class ID {class_id} exceeds class_names length in {img_file}")
                    continue
                orig_class = class_names[class_id]
                mapped_class = label_remap.get(orig_class, orig_class)
                if mapped_class == excluded_class:
                    continue
                if mapped_class == 'r':
                    board_class_count += 1
                target_split = 'valid' if image_counts['valid'][mapped_class] < 20 else 'train'
                tasks.append((img_path, label_path, output_path, target_split, mapped_class, 2 if mapped_class.islower() else 1, augment))
            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                for result, mapped_class, line_count in executor.map(process_image, tasks):
                    if result:
                        for target_split, cls, count in result:
                            image_counts[target_split][cls] += count
                            remapped_class_counts[cls] += count
                        original_class_counts[orig_class] += line_count
            gc.collect()
        logger.info(f"Original class counts (before remap): {dict(original_class_counts)}")
        logger.info(f"Images labeled as 'board' (remapped to 'r'): {board_class_count}")

    if os.path.exists(custom_dataset_path):
        available_files = os.listdir(custom_dataset_path)
        logger.info(f"Files in custom_dataset_path {custom_dataset_path}: {available_files}")
        for file_name in available_files:
            if not file_name.lower().endswith(('.jpg', '.png')):
                continue
            base_name = file_name.split('.')[0]
            target_class = file_mapping.get(file_name)
            if target_class is None:
                if len(base_name) >= 1 and base_name[0].lower() in ['b', 'k', 'n', 'p', 'q', 'r']:
                    target_class = base_name[0].lower() if base_name[0].islower() else base_name[0].upper()
                else:
                    continue
            if target_class not in remapped_class_names:
                continue
            custom_files_processed[target_class].append(file_name)
            if target_class == 'r':
                r_files_found.append(file_name)
            src_path = os.path.join(custom_dataset_path, file_name)
            if not os.path.exists(src_path):
                continue
            img = cv2.imread(src_path)
            if img is None:
                continue
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            if target_class.islower():
                img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
                img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            target_split = 'valid' if image_counts['valid'][target_class] < 20 else ('train' if random.random() < 0.8 else 'valid')
            save_name = f"custom_{file_name.rsplit('.', 1)[0]}.jpg"
            save_path = os.path.join(output_path, target_split, target_class, save_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, img)
            image_counts[target_split][target_class] += 1
            remapped_class_counts[target_class] += 1
            aug_count = 2 if target_class.islower() else 1
            for aug_idx in range(aug_count):
                augmented = augment(image=img)
                img_aug = augmented['image']
                save_name_aug = f"custom_{file_name.rsplit('.', 1)[0]}_aug{aug_idx}.jpg"
                save_path_aug = os.path.join(output_path, target_split, target_class, save_name_aug)
                cv2.imwrite(save_path_aug, img_aug)
                image_counts[target_split][target_class] += 1
                remapped_class_counts[target_class] += 1
            del img
            gc.collect()
        logger.info(f"Custom files processed per class: {dict(custom_files_processed)}")
        logger.info(f"Total custom files processed: {sum(len(files) for files in custom_files_processed.values())}")
        logger.info(f"Black rook ('r') files found in custom dataset: {r_files_found}")

    TARGET_IMAGES_PER_CLASS = 500  # Reduced for faster processing
    for split in ['train', 'valid']:
        for cls in remapped_class_names:
            cls_dir = os.path.join(output_path, split, cls)
            images = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.endswith('.jpg')]
            target_count = 20 if split == 'valid' else TARGET_IMAGES_PER_CLASS
            if len(images) < target_count:
                extra_images = random.choices(images, k=target_count - len(images))
                for i, src_path in enumerate(extra_images):
                    img = cv2.imread(src_path)
                    if img is None:
                        continue
                    save_name = f"oversample_{cls}_{i}.jpg"
                    save_path = os.path.join(cls_dir, save_name)
                    cv2.imwrite(save_path, img)
                    image_counts[split][cls] += 1
                    remapped_class_counts[cls] += 1
                    del img
                    gc.collect()
            elif len(images) > target_count:
                images = random.sample(images, target_count)
                for f in os.listdir(cls_dir):
                    if f.endswith('.jpg') and os.path.join(cls_dir, f) not in images:
                        os.remove(os.path.join(cls_dir, f))
                image_counts[split][cls] = len(images)

    logger.info(f"Remapped class counts (after processing): {dict(remapped_class_counts)}")
    logger.info(f"Final image counts - Train: {dict(image_counts['train'])}")
    logger.info(f"Final image counts - Valid: {dict(image_counts['valid'])}")
    return image_counts, remapped_class_names

def main():
    # Step 1: Install dependencies
    os.system("pip install -q ultralytics opencv-python numpy torch albumentations")

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
    class_ids = inspect_dataset(dataset_path, class_names)

    # Step 6: Preprocess to classification dataset
    output_path = "/content/chess_classification_dataset"
    image_counts, remapped_class_names = preprocess_dataset(dataset_path, output_path, class_names)

    # Step 7: Load classifier and train
    try:
        model = YOLO("yolov8n-cls.pt")  # Use nano model for faster training
        logger.info("YOLOv8n-cls model loaded.")
        device = 'cpu'
        logger.info(f"Training on device: {device}")
        model.train(
            data=output_path,
            epochs=50,  # Reduced epochs
            imgsz=224,
            batch=8,  # Increased batch size for CPU
            name="chess_piece_classifier",
            patience=5,
            device=device,
            optimizer="AdamW",
            lr0=0.0005,
            lrf=0.01,
            cos_lr=True,
            hsv_h=0.02,
            hsv_s=0.8,
            hsv_v=0.5,
            flipud=0.5,
            fliplr=0.5,
            mosaic=0.3,
            mixup=0.3
        )
        logger.info("Training completed.")

        # Step 8: Evaluate
        metrics = model.val()
        logger.info(f"Validation Top-1 Accuracy: {metrics.top1 * 100:.2f}%")
        logger.info(f"Validation Top-5 Accuracy: {metrics.top5 * 100:.2f}%")

        # Step 9: Evaluate white vs. black piece accuracy
        white_classes = ['B', 'K', 'N', 'P', 'Q', 'R']
        black_classes = ['b', 'k', 'n', 'p', 'q', 'r']
        white_correct, white_total = 0, 0
        black_correct, black_total = 0, 0
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        for cls in os.listdir(os.path.join(output_path, 'valid')):
            cls_dir = os.path.join(output_path, 'valid', cls)
            if not os.path.isdir(cls_dir):
                continue
            for img_file in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_file)
                results = model(img_path)
                pred_class = results[0].names[results[0].probs.top1]
                class_total[cls] += 1
                if pred_class == cls:
                    class_correct[cls] += 1
                    if cls in white_classes:
                        white_correct += 1
                        white_total += 1
                    elif cls in black_classes:
                        black_correct += 1
                        black_total += 1
                gc.collect()
        for cls in remapped_class_names:
            accuracy = (class_correct[cls] / class_total[cls] * 100) if class_total[cls] > 0 else 0
            logger.info(f"Class {cls}: Accuracy={accuracy:.2f}% ({class_correct[cls]}/{class_total[cls]})")
        logger.info(f"White piece accuracy: {white_correct/white_total*100:.2f}% ({white_correct}/{white_total})")
        logger.info(f"Black piece accuracy: {black_correct/black_total*100:.2f}% ({black_correct}/{black_total})")

        # Step 10: Export
        model.export(format="onnx")
        logger.info("Model exported to ONNX.")

    except Exception as e:
        logger.error(f"Training failed: {e}")

if __name__ == "__main__":
    main()
