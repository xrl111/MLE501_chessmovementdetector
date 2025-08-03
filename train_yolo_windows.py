"""
YOLO Training Script for Chess Piece Detection - Windows Compatible
"""
import os
import sys
import logging
from pathlib import Path
import torch
from ultralytics import YOLO
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def check_requirements():
    """Check if required packages are installed"""
    required_packages = {
        'torch': 'PyTorch',
        'ultralytics': 'Ultralytics YOLO',
        'yaml': 'PyYAML'
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name} installed")
        except ImportError:
            missing.append((package, name))
            print(f"✗ {name} missing")
    
    if missing:
        print("\nInstall missing packages:")
        for package, name in missing:
            if package == 'yaml':
                print(f"pip install pyyaml")
            else:
                print(f"pip install {package}")
        return False
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ CUDA not available, using CPU (training will be slower)")
    
    return True

def validate_dataset(data_yaml_path):
    """Validate dataset structure"""
    if not os.path.exists(data_yaml_path):
        logging.error(f"Dataset config not found: {data_yaml_path}")
        return False
    
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Check required fields
    required_fields = ['train', 'val', 'nc', 'names']
    for field in required_fields:
        if field not in data_config:
            logging.error(f"Missing field '{field}' in {data_yaml_path}")
            return False
    
    # Check paths exist
    base_path = Path(data_yaml_path).parent
    for split in ['train', 'val']:
        if split in data_config:
            split_path = base_path / data_config[split]
            if not split_path.exists():
                logging.error(f"Dataset path not found: {split_path}")
                return False
            print(f"✓ Found {split} data: {split_path}")
    
    print(f"✓ Dataset validated: {data_config['nc']} classes")
    print(f"  Classes: {data_config['names']}")
    
    return True

def train_model(data_yaml_path, model_size='n', epochs=100, batch_size=16, img_size=640):
    """Train YOLO model with Windows-specific configurations"""
    
    print(f"\n=== Starting YOLO Training ===")
    print(f"Model: YOLOv8{model_size}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}")
    print(f"Dataset: {data_yaml_path}")
    
    try:
        # Initialize model
        model = YOLO(f'yolov8{model_size}.pt')  # Load pre-trained model
        
        # Training parameters optimized for Windows
        train_args = {
            'data': data_yaml_path,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'patience': 50,  # Early stopping
            'save': True,
            'save_period': 10,  # Save every 10 epochs
            'cache': False,  # Disable caching on Windows to avoid memory issues
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'workers': 2,  # Reduced workers for Windows stability
            'project': 'chess_training',  # Project name
            'name': f'chess_yolov8{model_size}',  # Run name
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'SGD',
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'single_cls': False,
            'rect': False,  # Rectangular training
            'cos_lr': False,  # Cosine learning rate
            'close_mosaic': 10,  # Close mosaic augmentation in last N epochs
            'resume': False,
            'amp': True,  # Automatic Mixed Precision
            'fraction': 1.0,  # Use full dataset
            'profile': False,
            'freeze': None,  # Freeze layers
            'lr0': 0.01,  # Initial learning rate
            'lrf': 0.01,  # Final learning rate (lr0 * lrf)
            'momentum': 0.937,  # SGD momentum
            'weight_decay': 0.0005,  # Optimizer weight decay
            'warmup_epochs': 3.0,  # Warmup epochs
            'warmup_momentum': 0.8,  # Warmup initial momentum
            'warmup_bias_lr': 0.1,  # Warmup initial bias lr
            'box': 7.5,  # Box loss gain
            'cls': 0.5,  # Class loss gain
            'dfl': 1.5,  # DFL loss gain
            'pose': 12.0,  # Pose loss gain
            'kobj': 2.0,  # Keypoint object loss gain
            'label_smoothing': 0.0,  # Label smoothing epsilon
            'nbs': 64,  # Nominal batch size
            'overlap_mask': True,  # Masks should overlap during training
            'mask_ratio': 4,  # Mask downsample ratio
            'dropout': 0.0,  # Use dropout regularization
            'val': True,  # Validate/test during training
        }
        
        # Start training
        print("\n🚀 Starting training...")
        results = model.train(**train_args)
        
        # Training completed
        print(f"\n✅ Training completed!")
        print(f"Results saved in: chess_training/chess_yolov8{model_size}")
        
        # Export model
        print("\n📦 Exporting model...")
        model.export(format='onnx', dynamic=False, simplify=True)
        print(f"✅ ONNX model exported!")
        
        return results
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise

def main():
    """Main training function"""
    print("=== Chess Piece Detection Training (Windows) ===\n")
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Dataset path
    data_yaml_path = "yolo_dataset/data.yaml"
    
    # Check if dataset exists
    if not os.path.exists(data_yaml_path):
        print(f"❌ Dataset not found: {data_yaml_path}")
        print("Run: python convert_coco_to_yolo.py first")
        sys.exit(1)
    
    # Validate dataset
    if not validate_dataset(data_yaml_path):
        sys.exit(1)
    
    # Training configurations
    training_configs = [
        {
            'model_size': 'n',  # nano - fastest
            'epochs': 100,
            'batch_size': 16,
            'img_size': 640,
            'description': 'Fast training for testing'
        },
        # Uncomment for more thorough training:
        # {
        #     'model_size': 's',  # small - balanced
        #     'epochs': 200,
        #     'batch_size': 8,
        #     'img_size': 640,
        #     'description': 'Balanced speed/accuracy'
        # },
        # {
        #     'model_size': 'm',  # medium - best accuracy
        #     'epochs': 300,
        #     'batch_size': 4,
        #     'img_size': 640,
        #     'description': 'Best accuracy (slow)'
        # }
    ]
    
    # Train models
    for i, config in enumerate(training_configs, 1):
        print(f"\n=== Training Configuration {i}/{len(training_configs)} ===")
        print(f"Description: {config['description']}")
        
        try:
            results = train_model(
                data_yaml_path=data_yaml_path,
                model_size=config['model_size'],
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                img_size=config['img_size']
            )
            
            print(f"✅ Configuration {i} completed successfully!")
            
        except Exception as e:
            print(f"❌ Configuration {i} failed: {e}")
            continue
    
    print("\n🎉 All training completed!")
    print("\nNext steps:")
    print("1. Check results in chess_training/ directory")
    print("2. Use best.pt model in your detection script")
    print("3. Copy best.onnx to model_dataset/PieceDetection/")

if __name__ == "__main__":
    main()