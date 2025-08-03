"""
Convert COCO format dataset to YOLO format for Windows training
"""
import json
import os
from pathlib import Path
import shutil

def convert_coco_to_yolo(coco_file, output_dir, image_dir):
    """Convert COCO annotations to YOLO format"""
    
    # Load COCO annotations
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    labels_dir = os.path.join(output_dir, 'labels')
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    # Get categories
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    print(f"Found {len(categories)} categories: {list(categories.values())}")
    
    # Get images info
    images_info = {img['id']: img for img in coco_data['images']}
    
    # Process annotations
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # Convert each image
    converted_count = 0
    for img_id, img_info in images_info.items():
        img_filename = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Copy image file
        src_img_path = os.path.join(image_dir, img_filename)
        dst_img_path = os.path.join(images_dir, img_filename)
        if os.path.exists(src_img_path):
            shutil.copy2(src_img_path, dst_img_path)
        else:
            print(f"Warning: Image not found {src_img_path}")
            continue
        
        # Create YOLO label file
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_filename)
        
        with open(label_path, 'w') as f:
            if img_id in annotations_by_image:
                for ann in annotations_by_image[img_id]:
                    category_id = ann['category_id']
                    bbox = ann['bbox']  # [x, y, width, height] in pixels
                    
                    # Convert to YOLO format (normalized center x, center y, width, height)
                    x_center = (bbox[0] + bbox[2] / 2) / img_width
                    y_center = (bbox[1] + bbox[3] / 2) / img_height
                    width = bbox[2] / img_width
                    height = bbox[3] / img_height
                    
                    # Write YOLO format: class_id x_center y_center width height
                    f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        converted_count += 1
        if converted_count % 100 == 0:
            print(f"Converted {converted_count} images...")
    
    print(f"Conversion completed! Converted {converted_count} images.")
    return categories

def create_data_yaml(categories, train_dir, val_dir, test_dir, output_file):
    """Create data.yaml file for YOLO training"""
    
    # Get absolute paths for Windows
    train_path = os.path.abspath(train_dir).replace('\\', '/')
    val_path = os.path.abspath(val_dir).replace('\\', '/')
    test_path = os.path.abspath(test_dir).replace('\\', '/')
    
    yaml_content = f"""# Chess Piece Detection Dataset
path: {os.path.dirname(train_path).replace('\\', '/')}  # dataset root dir
train: {os.path.basename(train_path)}/images  # train images (relative to 'path')
val: {os.path.basename(val_path)}/images     # val images (relative to 'path')
test: {os.path.basename(test_path)}/images   # test images (relative to 'path')

# Classes
nc: {len(categories)}  # number of classes
names: {list(categories.values())}  # class names
"""
    
    with open(output_file, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created {output_file}")
    print("Classes mapping:")
    for cat_id, cat_name in categories.items():
        print(f"  {cat_id}: {cat_name}")

def main():
    """Main conversion function"""
    print("=== Chess Dataset COCO to YOLO Converter ===\n")
    
    # Setup directories
    base_dir = "."
    yolo_dataset_dir = "yolo_dataset"
    
    # Convert each split
    splits = ['train', 'valid', 'test']
    all_categories = None
    
    for split in splits:
        print(f"\n--- Converting {split} split ---")
        coco_file = os.path.join(base_dir, split, '_annotations.coco.json')
        image_dir = os.path.join(base_dir, split)
        output_dir = os.path.join(yolo_dataset_dir, split)
        
        if not os.path.exists(coco_file):
            print(f"Warning: {coco_file} not found, skipping {split}")
            continue
        
        categories = convert_coco_to_yolo(coco_file, output_dir, image_dir)
        
        if all_categories is None:
            all_categories = categories
        elif all_categories != categories:
            print("Warning: Categories differ between splits!")
    
    # Create data.yaml
    if all_categories:
        create_data_yaml(
            all_categories,
            os.path.join(yolo_dataset_dir, 'train'),
            os.path.join(yolo_dataset_dir, 'valid'),  
            os.path.join(yolo_dataset_dir, 'test'),
            os.path.join(yolo_dataset_dir, 'data.yaml')
        )
    
    print(f"\n=== Conversion Complete ===")
    print(f"YOLO dataset created in: {os.path.abspath(yolo_dataset_dir)}")
    print("Next steps:")
    print("1. Run: python train_yolo_windows.py")
    print("2. Check results in runs/detect/train/")

if __name__ == "__main__":
    main()