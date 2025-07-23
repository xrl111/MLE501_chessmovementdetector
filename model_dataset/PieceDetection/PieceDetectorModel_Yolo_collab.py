import os
import logging
import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import random
import torch
from albumentations import (
    Compose, Rotate, HorizontalFlip, ColorJitter, GaussNoise, RandomResizedCrop, RandomBrightnessContrast, RandomShadow, CLAHE, Sharpen, RandomContrast, RandomGamma
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_dataset(dataset_path, class_names, label_remap):
    """Inspect class IDs from detection labels and verify dataset balance."""
    class_ids = defaultdict(int)
    remapped_counts = defaultdict(int)
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
                    orig_class = class_names[class_id]
                    mapped_class = label_remap.get(orig_class, orig_class)
                    remapped_counts[mapped_class] += 1
                    if len(sample_labels) < 5:
                        sample_labels.append(f"{split}/{label_file}: {lines[0].strip()} (class {orig_class} -> {mapped_class})")
                except (ValueError, IndexError):
                    logger.warning(f"Invalid label format in {label_file}")
    logger.info(f"Original class ID distribution: {dict(class_ids)}")
    logger.info(f"Remapped class distribution: {dict(remapped_counts)}")
    logger.info("Sample labels:\n" + "\n".join(sample_labels))

    if class_ids:
        counts = np.array(list(remapped_counts.values()))
        imbalance_ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
        if imbalance_ratio > 3:
            logger.warning(f"Class imbalance detected (ratio: {imbalance_ratio:.2f}). Balancing in preprocessing.")
        target_count = int(np.median(counts))
        logger.info(f"Target images per class for balancing: {target_count}")
    else:
        target_count = 500
    return class_ids, target_count

def preprocess_dataset(dataset_path, output_path, class_names, custom_dataset_path, target_count=500):
    """Convert dataset to classification format, apply augmentation, and ensure class balance."""
    label_remap = {
        'B': 'b', 'K': 'board', 'N': 'k', 'P': 'n', 'Q': 'p', 'R': 'q',
        'b': 'B', 'k': 'K', 'n': 'N', 'p': 'P', 'q': 'Q', 'r': 'R',
        'board': 'r'
    }
    excluded_class = "excluded"
    remapped_class_names = sorted([c for c in class_names if c != 'board'])
    file_mapping = {
        'B.png': 'B', 'B_g.png': 'B',
        'K.png': 'K', 'K_g.png': 'K',
        'N.png': 'N', 'N_g.png': 'N',
        'P.png': 'P', 'P_g.png': 'P',
        'Q.png': 'Q', 'Q_g.png': 'Q',
        'R.png': 'R', 'R_g.png': 'R',
        'bb.png': 'b', 'bb_g.png': 'b',
        'kb.png': 'k', 'kb_g.png': 'k',
        'nb.png': 'n', 'nb_g.png': 'n',
        'pb.png': 'p', 'pb_g.png': 'p',
        'qb.png': 'q', 'qb_g.png': 'q',
        'rb.png': 'r', 'rb_g.png': 'r'
    }

    # Enhanced augmentation pipeline
    augment = Compose([
        Rotate(limit=45, p=0.9),
        HorizontalFlip(p=0.7),
        RandomResizedCrop(size=(224, 224), scale=(0.6, 1.0), p=0.8),
        ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.4, p=0.8),
        GaussNoise(p=0.6),
        RandomBrightnessContrast(p=0.8),
        RandomShadow(p=0.7),
        CLAHE(p=0.7),
        Sharpen(p=0.6),
        RandomContrast(p=0.7),
        RandomGamma(p=0.7)
    ])

    image_counts = {'train': defaultdict(int), 'valid': defaultdict(int)}
    temp_image_paths = {'train': defaultdict(list), 'valid': defaultdict(list)}
    original_class_counts = defaultdict(int)
    remapped_class_counts = defaultdict(int)

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
            for img_file in os.listdir(image_dir):
                if not img_file.lower().endswith(('.jpg', '.png')):
                    continue
                img_path = os.path.join(image_dir, img_file)
                label_path = os.path.join(label_dir, img_file.rsplit('.', 1)[0] + '.txt')
                if not os.path.exists(label_path):
                    continue
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                for i, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    try:
                        class_id, x_center, y_center, width, height = map(float, parts)
                        class_id = int(class_id)
                        if class_id >= len(class_names):
                            logger.warning(f"Class ID {class_id} exceeds class_names length in {img_file}")
                            continue
                        orig_class = class_names[class_id]
                        original_class_counts[orig_class] += 1
                        mapped_class = label_remap.get(orig_class, orig_class)
                        if mapped_class == excluded_class:
                            logger.debug(f"Skipping excluded class image {img_file} (original class: {orig_class})")
                            continue
                        if mapped_class == 'r':
                            board_class_count += 1
                        img = cv2.imread(img_path)
                        if img is None:
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
                            continue
                        cropped = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_AREA)
                        target_split = 'valid' if image_counts['valid'][mapped_class] < 200 else 'train'
                        save_name = f"orig_{img_file.rsplit('.', 1)[0]}_{i}.jpg"
                        save_path = os.path.join(output_path, target_split, mapped_class, save_name)
                        temp_image_paths[target_split][mapped_class].append((cropped, save_path))
                        image_counts[target_split][mapped_class] += 1
                        remapped_class_counts[mapped_class] += 1
                        logger.debug(f"Assigned original image {img_file} (class {orig_class} -> {mapped_class}) to {target_split}/{mapped_class}")
                        aug_count = 12 if mapped_class in ['b', 'B', 'board', 'k'] else 6
                        for aug_idx in range(aug_count):
                            augmented = augment(image=cropped)
                            img_aug = augmented['image']
                            save_name_aug = f"orig_{img_file.rsplit('.', 1)[0]}_{i}_aug{aug_idx}.jpg"
                            save_path_aug = os.path.join(output_path, target_split, mapped_class, save_name_aug)
                            temp_image_paths[target_split][mapped_class].append((img_aug, save_path_aug))
                            image_counts[target_split][mapped_class] += 1
                            remapped_class_counts[mapped_class] += 1
                    except ValueError as e:
                        logger.error(f"Invalid number format in {label_path}, line {i+1}: {e}")
                        continue
                    except IndexError as e:
                        logger.error(f"Malformed label in {label_path}, line {i+1}: {e}")
                        continue
        logger.info(f"Original class counts (before remap): {dict(original_class_counts)}")
        logger.info(f"Images labeled as 'board' (remapped to 'r'): {board_class_count}")

    custom_files_processed = 0
    required_classes = set(remapped_class_names)
    found_classes = set()
    missing_files = defaultdict(list)
    if os.path.exists(custom_dataset_path):
        available_files = os.listdir(custom_dataset_path)
        logger.info(f"Files in custom_dataset_path {custom_dataset_path}: {available_files}")
        for file_name in available_files:
            if not file_name.lower().endswith(('.jpg', '.png')):
                logger.warning(f"Skipping non-image file {file_name}")
                continue
            base_name = file_name.split('.')[0]
            target_class = None
            if file_name in file_mapping:
                target_class = file_mapping[file_name]
            elif len(base_name) >= 1 and base_name[0].lower() in ['b', 'k', 'n', 'p', 'q', 'r']:
                target_class = base_name[0].lower() if base_name[0].islower() else base_name[0].upper()
            if target_class not in remapped_class_names:
                logger.warning(f"Target class {target_class} not in remapped_class_names, skipping {file_name}")
                continue
            found_classes.add(target_class)
            src_path = os.path.join(custom_dataset_path, file_name)
            if not os.path.exists(src_path):
                logger.warning(f"Custom file {src_path} not found")
                continue
            img = cv2.imread(src_path)
            if img is None:
                logger.warning(f"Failed to load custom image {src_path}")
                continue
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            target_split = 'valid' if image_counts['valid'][target_class] < 200 else ('train' if random.random() < 0.8 else 'valid')
            save_name = f"custom_{file_name.rsplit('.', 1)[0]}.jpg"
            save_path = os.path.join(output_path, target_split, target_class, save_name)
            temp_image_paths[target_split][target_class].append((img, save_path))
            image_counts[target_split][target_class] += 1
            remapped_class_counts[target_class] += 1
            logger.debug(f"Assigned custom image {file_name} to {target_split}/{target_class}")
            aug_count = 12 if target_class in ['b', 'B', 'board', 'k'] else 6
            for aug_idx in range(aug_count):
                augmented = augment(image=img)
                img_aug = augmented['image']
                save_name_aug = f"custom_{file_name.rsplit('.', 1)[0]}_aug{aug_idx}.jpg"
                save_path_aug = os.path.join(output_path, target_split, target_class, save_name_aug)
                temp_image_paths[target_split][target_class].append((img_aug, save_path_aug))
                image_counts[target_split][target_class] += 1
                remapped_class_counts[target_class] += 1
            custom_files_processed += 1
        logger.info(f"Total custom files processed: {custom_files_processed}")
        missing_classes = required_classes - found_classes
        for cls in missing_classes:
            expected_files = [k for k, v in file_mapping.items() if v == cls]
            missing_files[cls].extend(expected_files)
        if missing_classes:
            logger.error(f"Custom dataset missing classes: {missing_classes}. Expected files: {dict(missing_files)}")
            return None, None
    else:
        logger.error(f"Custom dataset path {custom_dataset_path} does not exist")
        return None, None

    for split in ['train', 'valid']:
        for cls in remapped_class_names:
            images = temp_image_paths[split][cls]
            target_count_split = 200 if split == 'valid' else max(target_count, 2000 if cls in ['b', 'B', 'board', 'k'] else 1000)
            if images and image_counts[split][cls] < target_count_split:
                extra_images = random.choices(images, k=target_count_split - image_counts[split][cls])
                for i, (img, _) in enumerate(extra_images):
                    save_name = f"oversample_{cls}_{i}.jpg"
                    save_path = os.path.join(output_path, split, cls, save_name)
                    temp_image_paths[split][cls].append((img, save_path))
                    image_counts[split][cls] += 1
                    remapped_class_counts[cls] += 1
                    logger.debug(f"Oversampled image for {split}/{cls}: {save_name}")

    for split in ['train', 'valid']:
        for cls in remapped_class_names:
            images = temp_image_paths[split][cls]
            target_count_split = 200 if split == 'valid' else max(target_count, 2000 if cls in ['b', 'B', 'board', 'k'] else 1000)
            if len(images) > target_count_split:
                images = random.sample(images, target_count_split)
                logger.info(f"Reduced {split}/{cls} to {target_count_split} images")
            for img, save_path in images:
                cv2.imwrite(save_path, img)
                logger.debug(f"Saved {os.path.basename(save_path)} to {save_path}")
            image_counts[split][cls] = len(images)

    logger.info(f"Remapped class counts (after processing): {dict(remapped_class_counts)}")
    missing_valid_classes = [cls for cls in remapped_class_names if image_counts['valid'][cls] == 0]
    if missing_valid_classes:
        logger.warning(f"Missing classes in valid set: {missing_valid_classes}")
    logger.info(f"Final image counts - Train: {dict(image_counts['train'])}")
    logger.info(f"Final image counts - Valid: {dict(image_counts['valid'])}")
    white_classes = ['b', 'board', 'k', 'n', 'p', 'q']  # Remapped white pieces
    black_classes = ['B', 'K', 'N', 'P', 'Q', 'R']      # Remapped black pieces
    white_train_total = sum(image_counts['train'][cls] for cls in white_classes)
    black_train_total = sum(image_counts['train'][cls] for cls in black_classes)
    white_valid_total = sum(image_counts['valid'][cls] for cls in white_classes)
    black_valid_total = sum(image_counts['valid'][cls] for cls in black_classes)
    logger.info(f"Train set: White pieces (remapped): {white_train_total}, Black pieces (remapped): {black_train_total}")
    logger.info(f"Valid set: White pieces (remapped): {white_valid_total}, Black pieces (remapped): {black_valid_total}")
    return image_counts, remapped_class_names

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='/content', help='Path to main dataset')
    parser.add_argument('--custom_dataset_path', default='/content/drive/MyDrive/MSE24_HN/chess_dataset', help='Path to custom dataset')
    parser.add_argument('--output_path', default='/content/chess_classification_dataset', help='Path for preprocessed dataset')
    args = parser.parse_args()

    # Step 1: Install dependencies
    os.system("pip install -q ultralytics opencv-python numpy torch albumentations")

    # Step 2: Download dataset
    try:
        os.system('curl -L "https://universe.roboflow.com/ds/lEIeDLYdtb?key=ytHQpJZNeT" -o roboflow.zip')
        os.system('unzip -q roboflow.zip')
        os.system('rm roboflow.zip')
        dataset_path = args.dataset_path
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
    label_remap = {
        'B': 'b', 'K': 'board', 'N': 'k', 'P': 'n', 'Q': 'p', 'R': 'q',
        'b': 'B', 'k': 'K', 'n': 'N', 'p': 'P', 'q': 'Q', 'r': 'R',
        'board': 'r'
    }

    # Step 5: Inspect dataset
    class_ids, target_count = inspect_dataset(dataset_path, class_names, label_remap)
    if max(class_ids.keys(), default=-1) >= len(class_names):
        logger.error("Class IDs exceed available class_names list. Check alignment.")
        return

    # Step 6: Preprocess to classification dataset
    image_counts, remapped_class_names = preprocess_dataset(dataset_path, args.output_path, class_names, args.custom_dataset_path, target_count)
    if image_counts is None or remapped_class_names is None:
        logger.error("Preprocessing failed. Aborting.")
        return

    # Step 7: Final checks
    train_dir = os.path.join(args.output_path, 'train')
    valid_dir = os.path.join(args.output_path, 'valid')
    if not os.path.exists(train_dir) or not os.path.exists(valid_dir):
        logger.error("Preprocessed folders missing.")
        return
    if sum(image_counts['train'].values()) == 0 or sum(image_counts['valid'].values()) == 0:
        logger.error("No images after preprocessing. Aborting.")
        return
    missing_valid_classes = [cls for cls in remapped_class_names if image_counts['valid'][cls] == 0]
    if missing_valid_classes:
        logger.error(f"Validation set missing classes: {missing_valid_classes}. Cannot proceed with training.")
        return

    # Step 8: Load classifier and train
    try:
        model = YOLO("yolov8m-cls.pt")
        logger.info("YOLOv8m-cls model loaded.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    try:
        device = 0 if torch.cuda.is_available() else 'cpu'
        logger.info(f"Training on device: {device}")

        model.train(
            data=args.output_path,
            epochs=250,
            imgsz=224,
            batch=32,
            name="chess_piece_classifier",
            patience=40,
            device=device,
            optimizer="AdamW",
            lr0=0.0005,
            lrf=0.003,
            cos_lr=True,
            hsv_h=0.04,
            hsv_s=1.0,
            hsv_v=0.7,
            flipud=0.7,
            fliplr=0.7,
            mosaic=0.7,
            mixup=0.9,
            auto_augment='randaugment'
        )
        logger.info("Training completed.")

        # Step 9: Evaluate
        metrics = model.val()
        logger.info(f"Validation Top-1 Accuracy: {metrics.top1 * 100:.2f}%")
        logger.info(f"Validation Top-5 Accuracy: {metrics.top5 * 100:.2f}%")

        # Step 10: Evaluate white vs. black piece accuracy and per-class metrics
        white_classes = ['b', 'board', 'k', 'n', 'p', 'q']  # Remapped white pieces
        black_classes = ['B', 'K', 'N', 'P', 'Q', 'R']      # Remapped black pieces
        white_correct, white_total = 0, 0
        black_correct, black_total = 0, 0
        class_confidences = defaultdict(list)
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        confusion_matrix = defaultdict(lambda: defaultdict(int))
        misclassified_examples = []

        for split in ['valid']:
            for cls in os.listdir(os.path.join(args.output_path, split)):
                cls_dir = os.path.join(args.output_path, split, cls)
                if not os.path.isdir(cls_dir):
                    continue
                for img_file in os.listdir(cls_dir):
                    img_path = os.path.join(cls_dir, img_file)
                    results = model(img_path)
                    if not results or not hasattr(results[0], 'probs') or results[0].probs.top1 is None:
                        logger.error(f"Invalid prediction for {img_path}")
                        continue
                    confidence = results[0].probs.top1conf.item()
                    pred_class = results[0].names[results[0].probs.top1]
                    top3_indices = np.argsort(results[0].probs.data.cpu().numpy())[-3:][::-1]
                    top3_classes = [results[0].names[i] for i in top3_indices]
                    top3_probs = [results[0].probs.data[i].item() for i in top3_indices]
                    probs_dict = {results[0].names[i]: results[0].probs.data[i].item() for i in range(len(results[0].names))}
                    class_confidences[cls].append(confidence)
                    class_total[cls] += 1
                    logger.info(f"Image: {img_file}\nExpected Class: {cls}\nPredicted Class: {pred_class} (Confidence: {confidence:.2f})\nTop-3 Predictions: {', '.join([f'{c}: {p:.2f}' for c, p in zip(top3_classes, top3_probs)])}\nProbabilities: {', '.join([f'{k}: {v:.2f}' for k, v in probs_dict.items()])}\nCorrect: {pred_class == cls}")
                    if pred_class == cls:
                        class_correct[cls] += 1
                        if cls in white_classes:
                            white_correct += 1
                            white_total += 1
                        elif cls in black_classes:
                            black_correct += 1
                            black_total += 1
                    else:
                        confusion_matrix[cls][pred_class] += 1
                        misclassified_examples.append(f"{img_file}: true={cls}, pred={pred_class}, conf={confidence:.2f}, top3={', '.join([f'{c}: {p:.2f}' for c, p in zip(top3_classes, top3_probs)])}")

        for cls in remapped_class_names:
            accuracy = (class_correct[cls] / class_total[cls] * 100) if class_total[cls] > 0 else 0
            avg_conf = np.mean(class_confidences[cls]) if class_confidences[cls] else 0
            logger.info(f"Class {cls}: Accuracy={accuracy:.2f}% ({class_correct[cls]}/{class_total[cls]}), Avg Confidence={avg_conf:.2f}")

        if white_total > 0:
            logger.info(f"White piece accuracy (remapped): {white_correct/white_total*100:.2f}% ({white_correct}/{white_total})")
        else:
            logger.warning("No white piece images in validation set")
        if black_total > 0:
            logger.info(f"Black piece accuracy (remapped): {black_correct/black_total*100:.2f}% ({black_correct}/{black_total})")
        else:
            logger.warning("No black piece images in validation set")

        logger.info("Confusion Matrix (true -> predicted):")
        for true_cls in remapped_class_names:
            for pred_cls, count in confusion_matrix[true_cls].items():
                if count > 0:
                    logger.info(f"  {true_cls} misclassified as {pred_cls}: {count} times")

        if misclassified_examples:
            logger.info("Misclassified examples:")
            for example in misclassified_examples[:10]:
                logger.info(f"  {example}")
            if len(misclassified_examples) > 10:
                logger.info(f"  ... and {len(misclassified_examples) - 10} more")

        # Step 11: Export
        model.export(format="onnx")
        logger.info("Model exported to ONNX.")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return

if __name__ == "__main__":
    main()
