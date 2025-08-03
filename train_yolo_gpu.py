"""
YOLO Training Script for Chess Piece Detection - GPU Optimized for RTX 2060
"""
import os
import sys
import logging
import time
from pathlib import Path
import torch
from ultralytics import YOLO
import yaml
import gc

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def check_gpu_requirements():
    """Check GPU availability and memory"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"✅ GPU detected: {gpu_name}")
    print(f"✅ GPU memory: {gpu_memory:.1f} GB")
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    return True

def optimize_batch_size_for_gpu(gpu_memory_gb, img_size=640):
    """Calculate optimal batch size based on GPU memory"""
    
    # RTX 2060 6GB optimization
    if gpu_memory_gb >= 6:
        if img_size <= 640:
            return 32  # Aggressive for faster training
        elif img_size <= 800:
            return 16
        else:
            return 8
    elif gpu_memory_gb >= 4:
        return 16
    else:
        return 8

def train_model_gpu_optimized(data_yaml_path, model_size='n', epochs=150, custom_batch=None, img_size=640, early_stopping=True):
    """Train YOLO model optimized for GPU with early stopping"""
    
    # Check GPU
    if not check_gpu_requirements():
        raise RuntimeError("GPU not available")
    
    # Calculate optimal batch size
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if custom_batch:
        batch_size = custom_batch
    else:
        batch_size = optimize_batch_size_for_gpu(gpu_memory, img_size)
    
    # Early stopping configuration
    patience = 30 if early_stopping else epochs  # Stop if no improvement for 30 epochs
    min_epochs = 50  # Minimum epochs before early stopping can activate
    
    print(f"\n=== GPU Training Configuration ===")
    print(f"🎯 Model: YOLOv8{model_size}")
    print(f"🚀 Device: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
    print(f"📦 Batch size: {batch_size} (optimized)")
    print(f"🖼️ Image size: {img_size}")
    print(f"⏳ Max Epochs: {epochs}")
    print(f"📊 Mixed Precision: Enabled")
    print(f"⚡ Workers: 0 (Windows-compatible)")
    print(f"🛑 Early Stopping: {'Enabled' if early_stopping else 'Disabled'}")
    if early_stopping:
        print(f"   └─ Patience: {patience} epochs")
        print(f"   └─ Min epochs: {min_epochs}")
        print(f"   └─ Monitor: validation mAP@0.5")
    
    try:
        # Initialize model
        model = YOLO(f'yolov8{model_size}.pt')
        
        # GPU-optimized training parameters with early stopping
        train_args = {
            # Dataset
            'data': data_yaml_path,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            
            # GPU Optimization (Windows-compatible)
            'device': 0,  # Use GPU 0
            'workers': 0,  # Fix Windows multiprocessing issues
            'amp': True,  # Mixed precision training
            'cache': 'disk',  # Use disk cache for Windows stability
            'rect': False,  # Disable rect for Windows compatibility
            
            # Early Stopping & Monitoring
            'patience': patience,  # Early stopping patience
            'save': True,
            'save_period': 10,  # Save every 10 epochs
            'val': True,  # Enable validation
            'plots': True,  # Generate training plots
            
            # Project settings
            'project': 'chess_gpu_training',
            'name': f'chess_yolov8{model_size}_gpu',
            'exist_ok': True,
            
            # Model optimization
            'pretrained': True,
            'optimizer': 'AdamW',  # Better for GPU training
            'close_mosaic': 15,
            'cos_lr': True,  # Cosine learning rate
            
            # Learning rates optimized for GPU training
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # Loss weights
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            
            # Data augmentation (reduced for stability)
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
            
            # Validation
            'val': True,
            'split': 'val',
            'save_json': True,
            
            # Other
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'single_cls': False,
            'profile': False,
            'freeze': None,
        }
        
        print(f"\n🚀 Starting GPU training...")
        start_time = time.time()
        
        # Start training
        results = model.train(**train_args)
        
        training_time = time.time() - start_time
        print(f"\n✅ Training completed in {training_time/3600:.1f} hours!")
        
        # Export to ONNX for inference
        print(f"\n📦 Exporting to ONNX...")
        export_path = model.export(format='onnx', dynamic=False, simplify=True, opset=11)
        print(f"✅ ONNX model saved: {export_path}")
        
        # Copy to deployment location
        import shutil
        dest_path = "model_dataset/PieceDetection/best.onnx"
        shutil.copy2(export_path, dest_path)
        print(f"✅ Model deployed: {dest_path}")
        
        return results
        
    except Exception as e:
        logging.error(f"GPU training failed: {e}")
        print(f"\n💡 Troubleshooting tips:")
        print(f"1. Reduce batch_size if Out of Memory")
        print(f"2. Reduce img_size from {img_size} to 512")
        print(f"3. Set cache=False if RAM is insufficient")
        raise

def monitor_gpu_usage():
    """Monitor GPU usage during training"""
    if torch.cuda.is_available():
        print(f"\n📊 GPU Memory Status:")
        print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**3:.1f} GB")
        print(f"Cached: {torch.cuda.memory_reserved(0)/1024**3:.1f} GB")
        print(f"Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0))/1024**3:.1f} GB")

def main():
    """Main GPU training function"""
    print("🔥 Chess Piece Detection - GPU Training (RTX 2060 Optimized) 🔥\n")
    
    # Check dataset
    data_yaml_path = "yolo_dataset/data.yaml"
    if not os.path.exists(data_yaml_path):
        print(f"❌ Dataset not found: {data_yaml_path}")
        print("Run: python convert_coco_to_yolo.py first")
        sys.exit(1)
    
    # GPU Training configurations for RTX 2060 with Early Stopping
    gpu_configs = [
        {
            'name': 'Fast Training (Nano)',
            'model_size': 'n',
            'epochs': 150,
            'batch_size': 32,
            'img_size': 640,
            'description': '⚡ Fast training (1-2h) - Early stop enabled (patience: 30)'
        },
        {
            'name': 'Balanced Training (Small)', 
            'model_size': 's',
            'epochs': 200,
            'batch_size': 16,
            'img_size': 640,
            'description': '⚖️ Balanced training (3-4h) - Early stop enabled (patience: 30)'
        },
        {
            'name': 'High Quality (Medium)',
            'model_size': 'm',
            'epochs': 250,
            'batch_size': 8,
            'img_size': 640,
            'description': '🎯 Best accuracy (6-8h) - Early stop enabled (patience: 30)'
        }
    ]
    
    print("Available GPU training configurations:")
    for i, config in enumerate(gpu_configs, 1):
        print(f"{i}. {config['name']}")
        print(f"   {config['description']}")
        print(f"   Model: YOLOv8{config['model_size']}, Epochs: {config['epochs']}, Batch: {config['batch_size']}")
    
    print(f"\n0. Train all configurations sequentially")
    print(f"4. Custom early stopping configuration")
    
    try:
        choice = input(f"\nSelect configuration (0-{len(gpu_configs)+1}): ").strip()
        
        if choice == '0':
            # Train all configurations
            for i, config in enumerate(gpu_configs, 1):
                print(f"\n{'='*60}")
                print(f"🚀 Starting Configuration {i}/{len(gpu_configs)}: {config['name']}")
                print(f"{'='*60}")
                
                monitor_gpu_usage()
                
                results = train_model_gpu_optimized(
                    data_yaml_path=data_yaml_path,
                    model_size=config['model_size'],
                    epochs=config['epochs'],
                    custom_batch=config['batch_size'],
                    img_size=config['img_size'],
                    early_stopping=True
                )
                
                print(f"✅ Configuration {i} completed!")
                
                # Clear GPU cache between trainings
                torch.cuda.empty_cache()
                gc.collect()
                
        elif choice.isdigit() and 1 <= int(choice) <= len(gpu_configs):
            config = gpu_configs[int(choice) - 1]
            print(f"\n🚀 Starting: {config['name']}")
            
            monitor_gpu_usage()
            
            results = train_model_gpu_optimized(
                data_yaml_path=data_yaml_path,
                model_size=config['model_size'],
                epochs=config['epochs'],
                custom_batch=config['batch_size'],
                img_size=config['img_size'],
                early_stopping=True
            )
            
        elif choice == '4':
            print(f"\n🔧 Custom Early Stopping Configuration")
            print("Run: python train_with_custom_early_stopping.py")
            sys.exit(0)
            
        else:
            print("❌ Invalid choice")
            sys.exit(1)
        
        print(f"\n🎉 GPU Training Pipeline Completed!")
        print(f"\n📁 Results saved in: chess_gpu_training/")
        print(f"📁 Models deployed in: model_dataset/PieceDetection/")
        print(f"\n🔥 Next steps:")
        print(f"1. Test trained model: python chess_movement_detector_yolo.py")
        print(f"2. Check training metrics in TensorBoard")
        print(f"3. Validate on test set")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Training interrupted by user")
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()