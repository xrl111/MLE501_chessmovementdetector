"""
Quick Balanced Training for Windows - Fixed multiprocessing issues
"""
import os
import sys
import torch
from ultralytics import YOLO

def train_balanced_model():
    """Train YOLOv8s with balanced configuration optimized for Windows + RTX 2060"""
    
    print("🔥 Chess Detection - Balanced Training (Windows Optimized)")
    print("=" * 60)
    
    # Check dataset
    if not os.path.exists("yolo_dataset/data.yaml"):
        print("❌ Dataset not found. Run: python convert_coco_to_yolo.py")
        sys.exit(1)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        device = 0
    else:
        print("⚠️ Using CPU (slower)")
        device = 'cpu'
    
    print(f"\n🎯 Configuration:")
    print(f"   Model: YOLOv8s (balanced)")
    print(f"   Epochs: 200 (with early stopping)")
    print(f"   Batch: 16 (optimized for RTX 2060)")
    print(f"   Early Stop: 30 epochs patience")
    print(f"   Cache: disk (Windows stable)")
    print(f"   Workers: 0 (Windows multiprocessing fix)")
    
    # Initialize model
    model = YOLO('yolov8s.pt')
    
    # Windows-optimized training args
    train_args = {
        'data': 'yolo_dataset/data.yaml',
        'epochs': 200,
        'batch': 16,
        'imgsz': 640,
        'device': device,
        'workers': 0,  # Fix Windows multiprocessing
        'amp': True if device != 'cpu' else False,
        'cache': 'disk',  # Stable on Windows
        'rect': False,  # Disable for Windows compatibility
        'patience': 30,  # Early stopping
        'save': True,
        'save_period': 10,
        'val': True,
        'plots': True,
        'project': 'chess_training_windows',
        'name': 'balanced_yolov8s',
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
        'deterministic': True,
    }
    
    print(f"\n🚀 Starting training...")
    print(f"💡 Estimated time: 3-4 hours")
    print(f"📊 Monitor: watch GPU usage in Task Manager")
    
    try:
        # Start training
        results = model.train(**train_args)
        
        print(f"\n✅ Training completed!")
        
        # Export to ONNX
        print(f"📦 Exporting to ONNX...")
        export_path = model.export(format='onnx', dynamic=False, simplify=True)
        print(f"✅ ONNX exported: {export_path}")
        
        # Deploy model
        import shutil
        dest_path = "model_dataset/PieceDetection/best.onnx"
        os.makedirs("model_dataset/PieceDetection", exist_ok=True)
        shutil.copy2(export_path, dest_path)
        print(f"✅ Model deployed: {dest_path}")
        
        print(f"\n🎉 Training successful!")
        print(f"📁 Results: chess_training_windows/balanced_yolov8s/")
        print(f"🧪 Test model: python chess_movement_detector_yolo.py")
        
        return results
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Training stopped by user")
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        
        # Common solutions
        print(f"\n💡 Try these solutions:")
        print(f"1. Reduce batch size: 16 → 8")
        print(f"2. Reduce image size: 640 → 512") 
        print(f"3. Free up system memory")
        print(f"4. Check dataset integrity")

if __name__ == "__main__":
    train_balanced_model()