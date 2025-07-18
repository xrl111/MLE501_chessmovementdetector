import os
import shutil
import cv2
from tqdm import tqdm
from ultralytics import YOLO

# ------------------------------------------
# Configuration
# ------------------------------------------
DETECTION_DATASET_PATH = "/content/chess_detect_dataset"  # original detection dataset
CLASSIFICATION_DATASET_PATH = "/content/chess_classification_dataset"  # output classification dataset
IMAGE_SIZE = 224
MODEL_NAME = "yolov8n-cls.pt"
EPOCHS = 20
BATCH = 32

# Remapping logic
label_remap = {
    'B': 'b', 'K': 'board', 'N': 'k', 'P': 'n', 'Q': 'p', 'R': 'q',
    'b': 'B', 'board': 'r', 'k': 'K', 'n': 'N', 'p': 'P', 'q': 'Q', 'r': 'R'
}

# Final class names (after remapping)
class_names = sorted(set(label_remap.values()))

# ------------------------------------------
# Preprocessing Function
# ------------------------------------------
def preprocess_dataset(detection_path, classification_path, image_size):
    print("🧹 Preprocessing dataset...")
    if os.path.exists(classification_path):
        shutil.rmtree(classification_path)
    os.makedirs(classification_path)

    for split in ['train', 'valid']:
        image_dir = os.path.join(detection_path, split, 'images')
        label_dir = os.path.join(detection_path, split, 'labels')
        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            print(f"⚠️ Missing split: {split}")
            continue

        for class_label in class_names:
            os.makedirs(os.path.join(classification_path, split, class_label), exist_ok=True)

        for img_file in tqdm(os.listdir(image_dir), desc=f"Processing {split}"):
            if not img_file.endswith(('.jpg', '.png', '.jpeg')):
                continue

            image_path = os.path.join(image_dir, img_file)
            label_path = os.path.join(label_dir, img_file.rsplit('.', 1)[0] + ".txt")

            if not os.path.exists(label_path):
                continue

            with open(label_path, 'r') as f:
                lines = f.readlines()

            image = cv2.imread(image_path)
            if image is None:
                continue
            h, w, _ = image.shape

            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, x_center, y_center, bw, bh = map(float, parts)
                class_id = int(class_id)

                # Map original class ID to remapped label
                if class_id >= len(class_names):
                    continue

                # Get original label → remapped label
                original_label = class_names[class_id]
                target_cls = label_remap.get(original_label, original_label)

                # Convert normalized to pixel values
                cx, cy, bw, bh = x_center * w, y_center * h, bw * w, bh * h
                x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
                x2, y2 = int(cx + bw / 2), int(cy + bh / 2)

                # Clamp to image size
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, w - 1), min(y2, h - 1)

                crop = image[y1:y2, x1:x2]
                if crop.shape[0] < 10 or crop.shape[1] < 10:
                    continue

                resized = cv2.resize(crop, (image_size, image_size))
                out_name = f"{img_file.rsplit('.', 1)[0]}_{i}.jpg"
                out_path = os.path.join(classification_path, split, target_cls, out_name)
                cv2.imwrite(out_path, resized)

    print("✅ Preprocessing complete.")

# ------------------------------------------
# Training Function
# ------------------------------------------
def train_model(data_path, model_name, epochs, batch):
    print("🧠 Starting YOLOv8 classification training...")
    model = YOLO(model_name)

    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=IMAGE_SIZE,
        batch=batch
    )
    print("✅ Training complete.")
    return results

# ------------------------------------------
# Entry Point
# ------------------------------------------
if __name__ == "__main__":
    preprocess_dataset(DETECTION_DATASET_PATH, CLASSIFICATION_DATASET_PATH, IMAGE_SIZE)
    train_model(CLASSIFICATION_DATASET_PATH, MODEL_NAME, EPOCHS, BATCH)
