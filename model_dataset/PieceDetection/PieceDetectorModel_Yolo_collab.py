import os
import logging
import cv2
from collections import defaultdict
import numpy as np
import random
import gc
import requests
from datetime import datetime
import torch
from ultralytics import YOLO
from albumentations import Compose, ColorJitter, GaussNoise, RandomBrightnessContrast  # ✅ Fixed


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
INITIAL_IMAGE_LIMIT = 100  # Minimum number of images to process before balancing
MAX_FILES_PER_CLASS = 1000  # Maximum files per class

def inspect_dataset(dataset_path, class_names):
    """Inspect class IDs from detection labels and verify dataset balance.

    Args:
        dataset_path (str): Path to dataset with train/valid splits.
        class_names (list): List of class names in order of class IDs.

    Returns:
        dict: Class ID counts.
    """
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
                        sample_labels.append(f"{split}/{label_file}: {lines[0].strip()} (class {class_names[class_id]})")
                except (ValueError, IndexError):
                    logger.warning(f"Invalid label format in {label_file}")
    logger.info(f"Class ID distribution: {dict(class_ids)}")
    logger.info("Sample labels:\n" + "\n".join(sample_labels))
    return class_ids

def preprocess_dataset(dataset_path, output_path, class_names, custom_dataset_path="custom_dataset", max_files_per_class=MAX_FILES_PER_CLASS):
    """Convert dataset to classification format, apply augmentation, and enforce 80/20 split after 20 images.

    Args:
        dataset_path (str): Path to input dataset (train/valid splits with images and labels).
        output_path (str): Path to save processed classification dataset.
        class_names (list): List of class names in order of class IDs.
        custom_dataset_path (str): Path to custom images with predefined class mappings.
        max_files_per_class (int): Maximum number of images per class.

    Returns:
        tuple: (image_counts, remapped_class_names)
    """
    # Create a mapping of class names to unique folder names (e.g., 0_B, 1_K, 6_b, 8_k)
    class_to_folder = {cls: f"{cls}" for i, cls in enumerate(class_names)}
    
    # Explicitly exclude 'board' from processing
    remapped_class_names = [class_to_folder[c] for c in class_names if c != 'board']
    file_mapping = {k: class_to_folder[v] for k, v in {
        'K.png': 'K', 'N.png': 'N', 'B_g.png': 'B', 'N_g.png': 'N', 'Q_g.png': 'Q',
        'B.png': 'B', 'R.png': 'R', 'R_g.png': 'R', 'P_g.png': 'P', 'K_g.png': 'K',
        'P.png': 'P', 'Q.png': 'Q', 'pb_g.png': 'p', 'rb.png': 'r', 'kb.png': 'k',
        'qb.png': 'q', 'qb_g.png': 'q', 'bb_g.png': 'b', 'nb_g.png': 'n', 'rb_g.png': 'r',
        'kb_g.png': 'k', 'pb.png': 'p', 'bb.png': 'b', 'nb.png': 'n', 'custom_bb.png': 'b',
        'b_extra.png': 'b', 'b_extra2.png': 'b', 'custom_bb_g.png': 'b', 'custom_nb.png': 'n',
        'n_new1.png': 'n', 'n_new2.png': 'n', 'custom_nb_g.png': 'n', 'custom_kb.png': 'k',
        'k_extra.png': 'k', 'k_extra2.png': 'k', 'custom_kb_g.png': 'k', 'custom_pb.png': 'p',
        'p_new1.png': 'p', 'p_new2.png': 'p', 'custom_pb_g.png': 'p', 'custom_rb.png': 'r',
        'r_extra.png': 'r', 'r_extra2.png': 'r', 'custom_rb_g.png': 'r', 'custom_qb.png': 'q',
        'q_extra.png': 'q', 'q_extra2.png': 'q', 'custom_qb_g.png': 'q'
    }.items()}

    augment = Compose([
        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.05, p=0.5),
        GaussNoise(p=0.3), RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5)
    ])

    image_counts = {'train': defaultdict(int), 'valid': defaultdict(int)}
    original_class_counts = defaultdict(int)
    remapped_class_counts = defaultdict(int)
    custom_files_processed = defaultdict(list)
    r_files_found = []

    # Create output directories
    for split in ['train', 'valid']:
        for cls in remapped_class_names:
            os.makedirs(os.path.join(output_path, split, cls), exist_ok=True)

    # Check disk space
    import shutil
    disk_usage = shutil.disk_usage('/content/drive')
    if disk_usage.free < 5 * 1024 * 1024 * 1024:  # Less than 5GB free
        logger.warning(f"Low disk space on Google Drive: {disk_usage.free / (1024**3):.2f} GB free")

    if dataset_path and os.path.exists(dataset_path):
        processed_images = 0
        all_files_processed = False
        for split in ['train', 'valid']:
            image_dir = os.path.join(dataset_path, split, 'images')
            label_dir = os.path.join(dataset_path, split, 'labels')
            if not os.path.exists(image_dir) or not os.path.exists(label_dir):
                logger.warning(f"Missing image or label directory for {split}")
                continue
            image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))]
            for img_file in image_files:
                if all_files_processed or processed_images >= INITIAL_IMAGE_LIMIT:
                    break
                img_path = os.path.join(image_dir, img_file)
                label_path = os.path.join(label_dir, img_file.rsplit('.', 1)[0] + '.txt')
                if not os.path.exists(label_path):
                    continue
                # Load image once before processing lines
                img = cv2.imread(img_path)
                if img is None:
                    logger.error(f"Failed to load image {img_path}, skipping")
                    continue
                h, w = img.shape[:2]
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                for i, line in enumerate(lines):
                    if all_files_processed or processed_images >= INITIAL_IMAGE_LIMIT:
                        break
                    parts = line.strip().split()
                    if len(parts) != 5:
                        logger.warning(f"Invalid label format in {img_file}, line {i+1}")
                        continue
                    try:
                        class_id, x_center, y_center, width, height = map(float, parts)
                        class_id = int(class_id)
                        if class_id >= len(class_names):
                            logger.warning(f"Class ID {class_id} exceeds class_names in {img_file}")
                            continue
                        orig_class = class_names[class_id]
                        original_class_counts[orig_class] += 1
                        mapped_class = class_to_folder[orig_class]
                        if orig_class == 'board':
                            continue
                        # Enforce 80/20 split
                        total_class_count = image_counts['train'][mapped_class] + image_counts['valid'][mapped_class]
                        if total_class_count >= max_files_per_class:
                            continue
                        target_split = 'train'
                        if image_counts['train'][mapped_class] >= int(max_files_per_class * 0.8) and image_counts['valid'][mapped_class] < int(max_files_per_class * 0.2):
                            target_split = 'valid'
                        if mapped_class not in remapped_class_names:
                            logger.warning(f"Invalid mapped class '{mapped_class}' for {img_file}, line {i+1}, skipping")
                            continue
                        x_center *= w; y_center *= h; width *= w; height *= h
                        x1, y1 = int(max(x_center - width / 2, 0)), int(max(y_center - height / 2, 0))
                        x2, y2 = int(min(x_center + width / 2, w)), int(min(y_center + height / 2, h))
                        cropped = img[y1:y2, x1:x2]
                        if cropped.size == 0:
                            logger.warning(f"Empty crop for {img_file}, line {i+1}")
                            continue
                        cropped = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_AREA)
                        if orig_class.islower():
                            img_yuv = cv2.cvtColor(cropped, cv2.COLOR_BGR2YUV)
                            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
                            cropped = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
                        save_name = f"orig_{img_file.rsplit('.', 1)[0]}_{i}.jpg"
                        save_path = os.path.join(output_path, target_split, mapped_class, save_name)
                        cv2.imwrite(save_path, cropped)
                        image_counts[target_split][mapped_class] += 1
                        remapped_class_counts[mapped_class] += 1
                        aug_count = 2 if orig_class.islower() else 1
                        for aug_idx in range(aug_count):
                            if image_counts[target_split][mapped_class] >= max_files_per_class:
                                break
                            augmented = augment(image=cropped)
                            img_aug = augmented['image']
                            save_name_aug = f"orig_{img_file.rsplit('.', 1)[0]}_{i}_aug{aug_idx}.jpg"
                            save_path_aug = os.path.join(output_path, target_split, mapped_class, save_name_aug)
                            cv2.imwrite(save_path_aug, img_aug)
                            image_counts[target_split][mapped_class] += 1
                            remapped_class_counts[mapped_class] += 1
                        processed_images += 1
                        if processed_images % 100 == 0:  # Reduced verbosity for Colab
                            logger.info(f"Processed {processed_images} images")
                    except Exception as e:
                        logger.error(f"Error processing {img_file}, line {i+1}: {e}")
                    finally:
                        if 'cropped' in locals():
                            del cropped
                        gc.collect()
                if 'img' in locals():
                    del img
                    gc.collect()
        if processed_images > 0:
            logger.info(f"Total images processed: {processed_images}")

        # Balancing phase after reaching 20 images
        if processed_images >= INITIAL_IMAGE_LIMIT and not all_files_processed:
            for split in ['train', 'valid']:
                image_dir = os.path.join(dataset_path, split, 'images')
                label_dir = os.path.join(dataset_path, split, 'labels')
                if not os.path.exists(image_dir) or not os.path.exists(label_dir):
                    logger.warning(f"Missing image or label directory for {split}")
                    continue
                image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))]
                for img_file in image_files:
                    if all_files_processed:
                        break
                    img_path = os.path.join(image_dir, img_file)
                    label_path = os.path.join(label_dir, img_file.rsplit('.', 1)[0] + '.txt')
                    if not os.path.exists(label_path):
                        continue
                    img = cv2.imread(img_path)
                    if img is None:
                        logger.error(f"Failed to load image {img_path}, skipping")
                        continue
                    h, w = img.shape[:2]
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                    for i, line in enumerate(lines):
                        if all_files_processed:
                            break
                        parts = line.strip().split()
                        if len(parts) != 5:
                            logger.warning(f"Invalid label format in {img_file}, line {i+1}")
                            continue
                        try:
                            class_id, x_center, y_center, width, height = map(float, parts)
                            class_id = int(class_id)
                            if class_id >= len(class_names):
                                logger.warning(f"Class ID {class_id} exceeds class_names in {img_file}")
                                continue
                            orig_class = class_names[class_id]
                            mapped_class = class_to_folder[orig_class]
                            if orig_class == 'board':
                                continue
                            total_class_count = image_counts['train'][mapped_class] + image_counts['valid'][mapped_class]
                            if total_class_count >= max_files_per_class:
                                continue
                            target_split = 'train'
                            if image_counts['train'][mapped_class] >= int(max_files_per_class * 0.8) and image_counts['valid'][mapped_class] < int(max_files_per_class * 0.2):
                                target_split = 'valid'
                            if mapped_class not in remapped_class_names:
                                logger.warning(f"Invalid mapped class '{mapped_class}' for {img_file}, line {i+1}, skipping")
                                continue
                            x_center *= w; y_center *= h; width *= w; height *= h
                            x1, y1 = int(max(x_center - width / 2, 0)), int(max(y_center - height / 2, 0))
                            x2, y2 = int(min(x_center + width / 2, w)), int(min(y_center + height / 2, h))
                            cropped = img[y1:y2, x1:x2]
                            if cropped.size == 0:
                                logger.warning(f"Empty crop for {img_file}, line {i+1}")
                                continue
                            cropped = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_AREA)
                            if orig_class.islower():
                                img_yuv = cv2.cvtColor(cropped, cv2.COLOR_BGR2YUV)
                                img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
                                cropped = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
                            save_name = f"orig_{img_file.rsplit('.', 1)[0]}_{i}_bal.jpg"
                            save_path = os.path.join(output_path, target_split, mapped_class, save_name)
                            cv2.imwrite(save_path, cropped)
                            image_counts[target_split][mapped_class] += 1
                            remapped_class_counts[mapped_class] += 1
                            aug_count = 2 if orig_class.islower() else 1
                            for aug_idx in range(aug_count):
                                if image_counts[target_split][mapped_class] >= max_files_per_class:
                                    break
                                augmented = augment(image=cropped)
                                img_aug = augmented['image']
                                save_name_aug = f"orig_{img_file.rsplit('.', 1)[0]}_{i}_bal_aug{aug_idx}.jpg"
                                save_path_aug = os.path.join(output_path, target_split, mapped_class, save_name_aug)
                                cv2.imwrite(save_path_aug, img_aug)
                                image_counts[target_split][mapped_class] += 1
                                remapped_class_counts[mapped_class] += 1
                            # Check balancing
                            min_count = min(image_counts['train'][cls] + image_counts['valid'][cls] for cls in remapped_class_names)
                            max_count = max(image_counts['train'][cls] + image_counts['valid'][cls] for cls in remapped_class_names)
                            if min_count == max_count or max_count >= max_files_per_class:
                                all_files_processed = True
                                logger.info(f"Processed {sum(image_counts['train'].values()) + sum(image_counts['valid'].values())} files, all classes balanced (min: {min_count}, max: {max_count}), stopping")
                        except Exception as e:
                            logger.error(f"Error processing {img_file}, line {i+1}: {e}")
                        finally:
                            if 'cropped' in locals():
                                del cropped
                            gc.collect()
                    if 'img' in locals():
                        del img
                        gc.collect()

    if os.path.exists(custom_dataset_path) and not all_files_processed:
        processed_custom = 0
        for file_name in os.listdir(custom_dataset_path):
            if not file_name.lower().endswith(('.jpg', '.png')):
                continue
            if all_files_processed or processed_images >= INITIAL_IMAGE_LIMIT:
                break
            base_name = file_name.split('.')[0]
            target_class = file_mapping.get(file_name)
            if target_class is None:
                if len(base_name) >= 1 and base_name[0].lower() in ['b', 'k', 'n', 'p', 'q', 'r']:
                    target_class = class_to_folder[base_name[0].lower() if base_name[0].islower() else base_name[0].upper()]
                else:
                    logger.warning(f"Cannot infer class for {file_name}, skipping")
                    continue
            if target_class not in remapped_class_names:
                logger.warning(f"Target class {target_class} not in remapped_class_names, skipping")
                continue
            custom_files_processed[target_class].append(file_name)
            if target_class == class_to_folder['r']:
                r_files_found.append(file_name)
            src_path = os.path.join(custom_dataset_path, file_name)
            img = cv2.imread(src_path)
            if img is None:
                logger.warning(f"Failed to load custom image {src_path}")
                continue
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            if target_class.split('_')[1].islower():
                img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
                img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            target_split = 'train'
            if image_counts['train'][target_class] >= int(max_files_per_class * 0.8) and image_counts['valid'][target_class] < int(max_files_per_class * 0.2):
                target_split = 'valid'
            if image_counts[target_split][target_class] >= max_files_per_class:
                continue
            save_name = f"custom_{file_name.rsplit('.', 1)[0]}.jpg"
            save_path = os.path.join(output_path, target_split, target_class, save_name)
            cv2.imwrite(save_path, img)
            image_counts[target_split][target_class] += 1
            remapped_class_counts[target_class] += 1
            aug_count = 2 if target_class.split('_')[1].islower() else 1
            for aug_idx in range(aug_count):
                if image_counts[target_split][target_class] >= max_files_per_class:
                    break
                augmented = augment(image=img)
                img_aug = augmented['image']
                save_name_aug = f"custom_{file_name.rsplit('.', 1)[0]}_aug{aug_idx}.jpg"
                save_path_aug = os.path.join(output_path, target_split, target_class, save_name_aug)
                cv2.imwrite(save_path_aug, img_aug)
                image_counts[target_split][target_class] += 1
                remapped_class_counts[mapped_class] += 1
            processed_images += 1
            processed_custom += 1
            if processed_images % 100 == 0:  # Reduced verbosity for Colab
                logger.info(f"Processed {processed_images} images")
            del img
            gc.collect()
        if processed_custom > 0:
            logger.info(f"Total custom images processed: {processed_custom}")
        logger.info(f"Custom files processed: {dict(custom_files_processed)}")
        missing_classes = [cls for cls in remapped_class_names if cls not in custom_files_processed]
        if missing_classes:
            logger.warning(f"No custom files for classes: {missing_classes}")

    # Balancing phase for custom data if needed
    if processed_images >= INITIAL_IMAGE_LIMIT and not all_files_processed:
        for file_name in os.listdir(custom_dataset_path):
            if not file_name.lower().endswith(('.jpg', '.png')):
                continue
            if all_files_processed:
                break
            base_name = file_name.split('.')[0]
            target_class = file_mapping.get(file_name)
            if target_class is None:
                if len(base_name) >= 1 and base_name[0].lower() in ['b', 'k', 'n', 'p', 'q', 'r']:
                    target_class = class_to_folder[base_name[0].lower() if base_name[0].islower() else base_name[0].upper()]
                else:
                    logger.warning(f"Cannot infer class for {file_name}, skipping")
                    continue
            if target_class not in remapped_class_names:
                logger.warning(f"Target class {target_class} not in remapped_class_names, skipping")
                continue
            if image_counts['train'][target_class] + image_counts['valid'][target_class] >= max_files_per_class:
                continue
            src_path = os.path.join(custom_dataset_path, file_name)
            img = cv2.imread(src_path)
            if img is None:
                logger.warning(f"Failed to load custom image {src_path}")
                continue
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            if target_class.split('_')[1].islower():
                img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
                img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            target_split = 'train'
            if image_counts['train'][target_class] >= int(max_files_per_class * 0.8) and image_counts['valid'][target_class] < int(max_files_per_class * 0.2):
                target_split = 'valid'
            if image_counts[target_split][target_class] >= max_files_per_class:
                target_split = 'valid' if target_split == 'train' else 'train'
            save_name = f"custom_{file_name.rsplit('.', 1)[0]}_bal.jpg"
            save_path = os.path.join(output_path, target_split, target_class, save_name)
            cv2.imwrite(save_path, img)
            image_counts[target_split][target_class] += 1
            remapped_class_counts[target_class] += 1
            aug_count = 2 if target_class.split('_')[1].islower() else 1
            for aug_idx in range(aug_count):
                if image_counts[target_split][target_class] >= max_files_per_class:
                    break
                augmented = augment(image=img)
                img_aug = augmented['image']
                save_name_aug = f"custom_{file_name.rsplit('.', 1)[0]}_bal_aug{aug_idx}.jpg"
                save_path_aug = os.path.join(output_path, target_split, target_class, save_name_aug)
                cv2.imwrite(save_path_aug, img_aug)
                image_counts[target_split][target_class] += 1
                remapped_class_counts[target_class] += 1
            # Check balancing
            min_count = min(image_counts['train'][cls] + image_counts['valid'][cls] for cls in remapped_class_names)
            max_count = max(image_counts['train'][cls] + image_counts['valid'][cls] for cls in remapped_class_names)
            if min_count == max_count or max_count >= max_files_per_class:
                all_files_processed = True
                logger.info(f"Processed {sum(image_counts['train'].values()) + sum(image_counts['valid'].values())} files, all classes balanced (min: {min_count}, max: {max_count}), stopping")
            del img
            gc.collect()

    # Enforce 80/20 split in trimming
    logger.info(f"Before 80/20 enforcement - Train: {dict(image_counts['train'])}")
    logger.info(f"Before 80/20 enforcement - Valid: {dict(image_counts['valid'])}")
    for cls in remapped_class_names:
        train_count = image_counts['train'][cls]
        valid_count = image_counts['valid'][cls]
        total_count = train_count + valid_count
        if total_count > max_files_per_class:
            total_count = max_files_per_class
        train_target = int(total_count * 0.8)
        valid_target = total_count - train_target
        if train_count > train_target:
            excess = train_count - train_target
            cls_dir = os.path.join(output_path, 'train', cls)
            images = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.endswith('.jpg')]
            if len(images) > train_target:
                images_to_move = random.sample(images, excess)
                for img_path in images_to_move:
                    if not os.path.exists(img_path):
                        logger.warning(f"File {img_path} does not exist, skipping")
                        continue
                    valid_img_path = os.path.normpath(img_path.replace('train', 'valid'))
                    if os.path.exists(valid_img_path):
                        logger.warning(f"File {valid_img_path} already exists, skipping")
                        continue
                    os.makedirs(os.path.dirname(valid_img_path), exist_ok=True)
                    os.rename(img_path, valid_img_path)
                    image_counts['train'][cls] -= 1
                    image_counts['valid'][cls] += 1
        elif train_count < train_target and valid_count > 0:
            to_move = min(train_target - train_count, valid_count)
            cls_dir_valid = os.path.join(output_path, 'valid', cls)
            images = [os.path.join(cls_dir_valid, f) for f in os.listdir(cls_dir_valid) if f.endswith('.jpg')]
            if len(images) >= to_move:
                images_to_move = random.sample(images, to_move)
                for img_path in images_to_move:
                    if not os.path.exists(img_path):
                        logger.warning(f"File {img_path} does not exist, skipping")
                        continue
                    train_img_path = os.path.normpath(img_path.replace('valid', 'train'))
                    if os.path.exists(train_img_path):
                        logger.warning(f"File {train_img_path} already exists, skipping")
                        continue
                    os.makedirs(os.path.dirname(train_img_path), exist_ok=True)
                    os.rename(img_path, train_img_path)
                    image_counts['valid'][cls] -= 1
                    image_counts['train'][cls] += 1

    logger.info(f"Final counts - Train: {dict(image_counts['train'])}")
    logger.info(f"Final counts - Valid: {dict(image_counts['valid'])}")
    return image_counts, remapped_class_names

def download_dataset(dataset_path):
    """Download and unzip the Roboflow dataset with fallback.

    Args:
        dataset_path (str): Path to save the downloaded dataset.
    """
    url = "https://universe.roboflow.com/ds/lEIeDLYdtb?key=ytHQpJZNeT"
    try:
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        os.chdir(dataset_path)
        os.system('curl -L "' + url + '" -o roboflow.zip')
        os.system('unzip -q roboflow.zip')
        os.system('rm roboflow.zip')
        logger.info(f"Dataset downloaded and unzipped to {dataset_path}")
    except Exception as e:
        logger.error(f"curl failed: {e}")
        try:
            logger.info("Attempting download with requests as fallback...")
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open('roboflow.zip', 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                os.system('unzip -q roboflow.zip')
                os.system('rm roboflow.zip')
                logger.info(f"Dataset downloaded and unzipped to {dataset_path} via fallback")
            else:
                logger.error(f"Download failed with status code {response.status_code}")
                raise
        except Exception as e2:
            logger.error(f"Fallback download failed: {e2}")
            raise
    finally:
        os.chdir(os.path.dirname(dataset_path) or os.getcwd())

def main():
    """Main function to download, inspect, preprocess, and train the chess dataset."""
    # Set working directory to Google Drive
    working_dir = '/content/'
    os.makedirs(working_dir, exist_ok=True)
    dataset_path = os.path.join(working_dir, "dataset")
    output_path = "/content/chess_classification_dataset"
    custom_dataset_path = "/content/drive/MyDrive/MSE24_HN/chess_dataset"

    # Step 1: Download dataset
    download_dataset(dataset_path)

    # Step 2: Validate structure
    if not os.path.exists(os.path.join(dataset_path, 'train')) or not os.path.exists(os.path.join(dataset_path, 'valid')):
        logger.error("Missing 'train' or 'valid' folders after download")
        return

    # Step 3: Define class names (order is important for correct piece mapping)
    class_names = ['B', 'K', 'N', 'P', 'Q', 'R', 'b', 'board', 'k', 'n', 'p', 'q', 'r']
    
    # Step 4: Inspect dataset
    inspect_dataset(dataset_path, class_names)

    # Step 5: Preprocess to classification dataset with file limit per class
    image_counts, remapped_class_names = preprocess_dataset(dataset_path, output_path, class_names, custom_dataset_path, max_files_per_class=MAX_FILES_PER_CLASS)

    # Step 6: Final checks
    train_dir = os.path.join(output_path, 'train')
    valid_dir = os.path.join(output_path, 'valid')
    if not os.path.exists(train_dir) or not os.path.exists(valid_dir):
        logger.error("Preprocessed folders missing.")
        return
    if sum(image_counts['train'].values()) == 0 or sum(image_counts['valid'].values()) == 0:
        logger.error("No images after preprocessing.")
        return

    # Step 7: Create data.yaml for YOLO training
    
    # Step 8: Load classifier and train
    try:
        model = YOLO("yolov8m-cls.pt")
        logger.info("YOLOv8m-cls model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading YOLOv8m-cls model: {e}")
        return

    try:
        # Determine device (GPU or CPU)
        if torch.cuda.is_available():
            device = 0  # Colab typically has a single GPU (T4)
            logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            logger.warning("No GPU detected, falling back to CPU. Training may be slower.")
        logger.info(f"Training on device: {device}")

        # Train the model
        start_time = datetime.now()
        results = model.train(
            data=output_path,
            epochs=30,
            imgsz=224,
            batch=16,  # Suitable for Colab T4 GPU
            name="chess_piece_classifier",
            patience=10,
            device=device,
            optimizer="AdamW",
            lr0=0.0001,
            lrf=0.01,
            cos_lr=True,
            dropout=0.3,
            weight_decay=0.0005,
            hsv_h=0.02,
            hsv_s=0.5,
            hsv_v=0.3,
            flipud=0.5,
            fliplr=0.5,
            mosaic=0.2,
            mixup=0.2,
            verbose=False,
            project=os.path.join(working_dir, "runs/train"),
            exist_ok=True
        )
        training_time = datetime.now() - start_time
        logger.info(f"Training completed in {training_time.total_seconds() / 60:.2f} minutes. Results saved to {os.path.join(working_dir, 'runs/train', 'chess_piece_classifier')}")

         # Step 9: Evaluate
        metrics = model.val()
        logger.info(f"Validation Top-1 Accuracy: {metrics.top1 * 100:.2f}%")
        logger.info(f"Validation Top-5 Accuracy: {metrics.top5 * 100:.2f}%")

        # Export model to ONNX
        export_path = os.path.join(working_dir, 'runs/train/chess_piece_classifier/best.onnx')
        model.export(format='onnx', path=export_path)
        logger.info(f"Model exported to {export_path}")
    except RuntimeError as e:
        logger.error(f"Training error (possible GPU memory issue): {e}")
        logger.info("Try reducing batch size (e.g., batchndx=8) or switching to CPU (device='cpu').")
        return
    except Exception as e:
        logger.error(f"Error during training or evaluation: {e}")
        return

    logger.info("Dataset downloaded, preprocessed, and trained successfully.")

if __name__ == "__main__":
    main()
