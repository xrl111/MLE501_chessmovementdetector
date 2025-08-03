"""
YOLO Training with Advanced Early Stopping Options
Allows customization of early stopping parameters
"""
import os
import sys
import torch
from ultralytics import YOLO
from training_monitor import TrainingMonitor

def get_early_stopping_config():
    """Get early stopping configuration from user"""
    
    print("\n🛑 Early Stopping Configuration")
    print("=" * 40)
    
    # Enable/disable early stopping
    enable_es = input("Enable early stopping? (y/N): ").strip().lower()
    if enable_es != 'y':
        return {'enabled': False}
    
    print("\n📊 Early Stopping Parameters:")
    
    # Patience (epochs to wait)
    patience = input("Patience (epochs to wait for improvement, default: 30): ").strip()
    patience = int(patience) if patience.isdigit() else 30
    
    # Minimum delta (minimum improvement)
    min_delta = input("Minimum delta (minimum improvement, default: 0.001): ").strip()
    min_delta = float(min_delta) if min_delta.replace('.', '').isdigit() else 0.001
    
    # Minimum epochs before early stopping
    min_epochs = input("Minimum epochs before early stopping (default: 50): ").strip()
    min_epochs = int(min_epochs) if min_epochs.isdigit() else 50
    
    # Metric to monitor
    print("\nMetric to monitor:")
    print("1. mAP@0.5 (recommended)")
    print("2. mAP@0.5:0.95")
    print("3. Validation loss")
    metric_choice = input("Choice (1-3, default: 1): ").strip()
    
    metric_map = {
        '1': 'mAP50',
        '2': 'mAP50_95', 
        '3': 'val_loss'
    }
    monitor_metric = metric_map.get(metric_choice, 'mAP50')
    
    # Mode (min or max)
    mode = 'max' if monitor_metric in ['mAP50', 'mAP50_95'] else 'min'
    
    # Save best model
    save_best = input("Save best model during training? (Y/n): ").strip().lower()
    save_best = save_best != 'n'
    
    # Restore best weights
    restore_best = input("Restore best weights after early stopping? (Y/n): ").strip().lower()
    restore_best = restore_best != 'n'
    
    config = {
        'enabled': True,
        'patience': patience,
        'min_delta': min_delta,
        'min_epochs': min_epochs,
        'monitor': monitor_metric,
        'mode': mode,
        'save_best': save_best,
        'restore_best': restore_best
    }
    
    print(f"\n✅ Early Stopping Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    return config

def train_with_custom_early_stopping():
    """Train with custom early stopping configuration"""
    
    print("🎯 YOLO Training with Custom Early Stopping")
    print("=" * 50)
    
    # Check dataset
    data_yaml = "yolo_dataset/data.yaml"
    if not os.path.exists(data_yaml):
        print(f"❌ Dataset not found: {data_yaml}")
        print("Run: python convert_coco_to_yolo.py first")
        sys.exit(1)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, using CPU (will be slower)")
        device = 'cpu'
    else:
        device = 0
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Get training configuration
    print("\n🔧 Training Configuration")
    print("-" * 30)
    
    model_size = input("Model size (n/s/m/l/x, default: n): ").strip() or 'n'
    epochs = input("Maximum epochs (default: 200): ").strip()
    epochs = int(epochs) if epochs.isdigit() else 200
    
    batch_size = input("Batch size (default: 16): ").strip()
    batch_size = int(batch_size) if batch_size.isdigit() else 16
    
    img_size = input("Image size (default: 640): ").strip()
    img_size = int(img_size) if img_size.isdigit() else 640
    
    # Get early stopping configuration
    es_config = get_early_stopping_config()
    
    # Training parameters
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'device': device,
        'workers': 8 if device != 'cpu' else 4,
        'amp': True if device != 'cpu' else False,
        'cache': True,
        'rect': True,
        'save': True,
        'save_period': 10,
        'val': True,
        'plots': True,
        'project': 'chess_custom_training',
        'name': f'yolov8{model_size}_custom_es',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'cos_lr': True,
        'close_mosaic': 15,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'verbose': True,
        'seed': 42,
    }
    
    # Apply early stopping configuration
    if es_config['enabled']:
        train_args['patience'] = es_config['patience']
        print(f"\n🛑 Early stopping enabled with patience: {es_config['patience']}")
    else:
        train_args['patience'] = epochs  # Disable early stopping
        print(f"\n⏩ Early stopping disabled")
    
    print(f"\n🚀 Starting training with YOLOv8{model_size}...")
    print(f"📊 Monitor training: python training_monitor.py")
    
    try:
        # Initialize model
        model = YOLO(f'yolov8{model_size}.pt')
        
        # Start training
        results = model.train(**train_args)
        
        print(f"\n✅ Training completed!")
        
        # Export to ONNX
        print(f"📦 Exporting to ONNX...")
        export_path = model.export(format='onnx', dynamic=False, simplify=True)
        
        # Deploy model
        import shutil
        dest_path = "model_dataset/PieceDetection/best.onnx"
        shutil.copy2(export_path, dest_path)
        print(f"✅ Model deployed: {dest_path}")
        
        return results
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise

def main():
    """Main function"""
    try:
        results = train_with_custom_early_stopping()
        
        print(f"\n🎉 Training Pipeline Completed!")
        print(f"📁 Results in: chess_custom_training/")
        print(f"🔍 Analyze results: python training_monitor.py")
        print(f"🧪 Test model: python chess_movement_detector_yolo.py")
        
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    main()