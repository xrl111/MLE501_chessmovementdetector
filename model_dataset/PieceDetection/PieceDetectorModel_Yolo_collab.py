import os
import logging
import shutil
import cv2
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_dataset(dataset_path):
    """Inspect class IDs from detection labels."""
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
    logger.info(f"Unique class IDs: {dict(class_ids)}")
    logger.info("Sample labels:\n" + "\n".join(sample_labels))
    return class_ids

def preprocess_dataset(dataset_path, output_path, class_names):
    """Convert detection dataset to classification dataset, with remapped labels."""
    label_remap = {
        "B": "b", "K": "board", "N": "k", "P": "n", "Q": "p", "R": "q",
        "b": "B", "board": "r", "k": "K", "n": "N", "p": "P", "q": "Q", "r": "R"
    }
    remapped_class_names = sorted(set(label_remap.values()))
    
    for split in ['train', 'valid']:
        for cls in remapped_class_names:
            os.makedirs(os.path.join(output_path, split, cls), exist_ok=True)

    for split in ['train', 'valid']:
        image_dir = os.path.join(dataset_path, split, 'images')
        label_dir = os.path.join(dataset_path, split, 'labels')

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

                class_id, x_center, y_center, width, height = map(float, parts)
                class_id = int(class_id)
                if class_id >= len(class_names):
                    continue

                orig_class = class_names[class_id]
                mapped_class = label_remap.get(orig_class, orig_class)

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

                save_name = f"{img_file.rsplit('.', 1)[0]}_{i}.jpg"
                save_path = os.path.join(output_path, split, mapped_class, save_name)
                cv2.imwrite(save_path, cropped)


def main():
    # Step 1: Install dependencies
    os.system("pip install -q ultralytics opencv-python")

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

    # Step 4: Define class names (label index must match order)
    class_names = ['b', 'k', 'n', 'p', 'q', 'r', 'B', 'K', 'N', 'P', 'Q', 'R', 'board']  # 13 classes

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
        model.train(
            data=output_path,  # DO NOT use data.yaml
            epochs=50,
            imgsz=224,
            batch=16,
            name="chess_piece_classifier",
            patience=10,
            device=0,  # GPU
            optimizer="Adam",
            lr0=0.001
        )
        logger.info("Training completed.")

        # Step 9: Evaluate
        metrics = model.val()
        logger.info(f"Validation Top-1 Accuracy: {metrics.top1 * 100:.2f}%")
        logger.info(f"Validation Top-5 Accuracy: {metrics.top5 * 100:.2f}%")

        # Step 10: Export
        model.export(format="onnx")
        logger.info("Model exported to ONNX.")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return

if __name__ == "__main__":
    main()
