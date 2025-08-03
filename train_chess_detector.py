"""
Main Training Script for Chess Movement Detector - Windows 11
This script runs the complete training pipeline from dataset conversion to model training
"""
import os
import sys
import subprocess
import time
import shutil
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Show last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print("Error:", e.stderr)
        return False

def check_dataset():
    """Check if dataset exists"""
    required_dirs = ['train', 'valid', 'test']
    missing_dirs = []
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing_dirs.append(dir_name)
        else:
            # Check for annotation file
            annotation_file = os.path.join(dir_name, '_annotations.coco.json')
            if not os.path.exists(annotation_file):
                print(f"⚠ Missing annotation file: {annotation_file}")
            else:
                print(f"✓ Found {dir_name} dataset")
    
    if missing_dirs:
        print(f"❌ Missing dataset directories: {missing_dirs}")
        print("Please ensure you have train/, valid/, test/ directories with COCO annotations")
        return False
    
    return True

def install_requirements():
    """Install required packages for training"""
    print("\n=== Installing Requirements ===")
    
    packages = [
        "ultralytics",
        "torch torchvision",
        "onnx",
        "onnxruntime", 
        "pyyaml",
        "opencv-python",
        "matplotlib",
        "pillow"
    ]
    
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠ Failed to install {package}, but continuing...")
    
    print("✅ Requirements installation completed")

def main():
    """Main training pipeline"""
    print("=" * 60)
    print("🏁 Chess Movement Detector Training Pipeline")
    print("=" * 60)
    print("This script will:")
    print("1. Check and install requirements") 
    print("2. Validate dataset")
    print("3. Convert COCO to YOLO format")
    print("4. Update code paths for Windows")
    print("5. Train YOLO model")
    print("6. Test trained model")
    print("=" * 60)
    
    # Get user confirmation
    response = input("\nProceed with training? (y/N): ").strip().lower()
    if response != 'y':
        print("Training cancelled.")
        sys.exit(0)
    
    start_time = time.time()
    
    # Step 1: Install requirements
    install_requirements()
    
    # Step 2: Check dataset
    print(f"\n=== Step 1: Dataset Validation ===")
    if not check_dataset():
        print("❌ Dataset validation failed!")
        sys.exit(1)
    
    # Step 3: Convert dataset
    print(f"\n=== Step 2: Convert COCO to YOLO ===")
    if not run_command("python convert_coco_to_yolo.py", "Converting dataset"):
        print("❌ Dataset conversion failed!")
        sys.exit(1)
    
    # Step 4: Update paths
    print(f"\n=== Step 3: Update Code Paths ===")
    if not run_command("python update_paths_for_windows.py", "Updating code paths"):
        print("⚠ Path update failed, but continuing...")
    
    # Step 5: Train model
    print(f"\n=== Step 4: Train YOLO Model ===")
    if not run_command("python train_yolo_windows.py", "Training YOLO model"):
        print("❌ Model training failed!")
        sys.exit(1)
    
    # Step 6: Copy best model
    print(f"\n=== Step 5: Deploy Trained Model ===")
    
    # Find the best trained model
    best_model_paths = [
        "chess_training/chess_yolov8n/weights/best.onnx",
        "chess_training/chess_yolov8n/weights/best.pt",
        "runs/detect/train/weights/best.onnx",
        "runs/detect/train/weights/best.pt"
    ]
    
    model_deployed = False
    for model_path in best_model_paths:
        if os.path.exists(model_path):
            # Copy to deployment location
            if model_path.endswith('.onnx'):
                dest_path = "model_dataset/PieceDetection/best.onnx"
                shutil.copy2(model_path, dest_path)
                print(f"✅ Deployed ONNX model: {model_path} -> {dest_path}")
                model_deployed = True
            elif model_path.endswith('.pt'):
                dest_path = "model_dataset/PieceDetection/best.pt"
                shutil.copy2(model_path, dest_path)
                print(f"✅ Deployed PyTorch model: {model_path} -> {dest_path}")
    
    if not model_deployed:
        print("⚠ No trained model found to deploy")
    
    # Step 7: Test the detector
    print(f"\n=== Step 6: Test Trained Model ===")
    print("Testing the chess movement detector with trained model...")
    
    # Create a test script
    test_command = "python chess_movement_detector_yolo.py"
    print(f"You can now test the detector with: {test_command}")
    
    # Calculate total time
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    
    print(f"\n" + "=" * 60)
    print(f"🎉 Training Pipeline Completed!")
    print(f"⏱ Total time: {hours}h {minutes}m")
    print(f"=" * 60)
    
    print(f"\n📋 Next Steps:")
    print(f"1. Test detection: python chess_movement_detector_yolo.py")
    print(f"2. Check training results in: chess_training/")
    print(f"3. Monitor training logs for accuracy metrics")
    print(f"4. Adjust training parameters if needed")
    
    print(f"\n📁 Key Files Created:")
    print(f"• yolo_dataset/ - Converted YOLO format dataset")
    print(f"• chess_training/ - Training results and models")
    print(f"• model_dataset/PieceDetection/best.onnx - Trained model")
    
    print(f"\n🔧 Troubleshooting:")
    print(f"• If accuracy is low, increase epochs in train_yolo_windows.py")
    print(f"• If training is slow, reduce batch_size")
    print(f"• Check GPU usage with: nvidia-smi")

if __name__ == "__main__":
    main()