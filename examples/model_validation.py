"""
Simple test script to verify our trained model works correctly
"""
import cv2
import os
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

def test_model_on_dataset():
    """Test our trained model on dataset images"""
    
    model_path = "model_dataset/PieceDetection/best.pt"
    test_images_dir = "yolo_dataset/test/images"
    
    print("🧪 Testing Trained Model")
    print("=" * 50)
    
    # Load model
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    try:
        model = YOLO(model_path)
        print(f"✅ Model loaded: {model_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Get test images
    if not os.path.exists(test_images_dir):
        print(f"❌ Test images not found: {test_images_dir}")
        return
    
    test_images = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.png'))]
    print(f"📁 Found {len(test_images)} test images")
    
    if not test_images:
        print("❌ No test images found")
        return
    
    # Test on first few images
    os.makedirs("model_test_results", exist_ok=True)
    
    for i, img_file in enumerate(test_images[:5]):  # Test first 5 images
        img_path = os.path.join(test_images_dir, img_file)
        print(f"\n🖼️ Testing: {img_file}")
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Failed to load image")
            continue
            
        print(f"   📏 Image size: {img.shape}")
        
        # Run inference
        try:
            results = model.predict(img, verbose=False, conf=0.3)
            
            if results and len(results) > 0:
                detections = results[0].boxes
                if detections is not None and len(detections) > 0:
                    print(f"   🎯 Found {len(detections)} detections")
                    
                    # Get class names
                    class_names = ['B', 'K', 'N', 'P', 'Q', 'R', 'b', 'k', 'n', 'p', 'q', 'r', 'empty']
                    
                    # Print detections
                    for j, (conf, cls) in enumerate(zip(detections.conf.cpu().numpy(), detections.cls.cpu().numpy())):
                        class_name = class_names[int(cls)] if int(cls) < len(class_names) else 'unknown'
                        print(f"     Detection {j+1}: {class_name} (conf: {conf:.3f})")
                    
                    # Save result image
                    result_img = results[0].plot()
                    result_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
                    output_path = f"model_test_results/test_{i+1}_{img_file}"
                    cv2.imwrite(output_path, result_img_bgr)
                    print(f"   💾 Saved: {output_path}")
                    
                else:
                    print(f"   ⚪ No detections found")
            else:
                print(f"   ❌ No results from model")
                
        except Exception as e:
            print(f"   ❌ Inference failed: {e}")
    
    print(f"\n✅ Model test completed!")
    print(f"📁 Results saved in: model_test_results/")

def test_model_on_single_square():
    """Test model on a single extracted square"""
    
    model_path = "model_dataset/PieceDetection/best.pt"
    
    # Create a simple test square (64x64 white image with black circle = piece)
    test_square = np.ones((64, 64, 3), dtype=np.uint8) * 255  # White background
    cv2.circle(test_square, (32, 32), 20, (0, 0, 0), -1)  # Black circle (piece)
    
    print("\n🔬 Testing on synthetic square")
    print("=" * 40)
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    try:
        model = YOLO(model_path)
        print(f"✅ Model loaded")
        
        # Test prediction
        results = model.predict(test_square, verbose=False, conf=0.1)
        
        if results and len(results) > 0 and len(results[0].boxes) > 0:
            detections = results[0].boxes
            print(f"🎯 Detections: {len(detections)}")
            
            class_names = ['B', 'K', 'N', 'P', 'Q', 'R', 'b', 'k', 'n', 'p', 'q', 'r', 'empty']
            for conf, cls in zip(detections.conf.cpu().numpy(), detections.cls.cpu().numpy()):
                class_name = class_names[int(cls)] if int(cls) < len(class_names) else 'unknown'
                print(f"   {class_name}: {conf:.3f}")
        else:
            print("⚪ No detections")
            
        # Save test square
        cv2.imwrite("model_test_results/synthetic_square.png", test_square)
        print("💾 Synthetic square saved")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

def analyze_model_info():
    """Print model information"""
    
    model_path = "model_dataset/PieceDetection/best.pt"
    
    print("\n📊 Model Analysis")
    print("=" * 30)
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    try:
        model = YOLO(model_path)
        
        print(f"Model type: {type(model)}")
        print(f"Model task: {getattr(model, 'task', 'unknown')}")
        
        # Print model info
        if hasattr(model, 'model') and hasattr(model.model, 'yaml'):
            print(f"Model config: {model.model.yaml}")
            
        if hasattr(model, 'names'):
            print(f"Classes: {model.names}")
            
        # Model size
        model_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Model size: {model_size:.1f} MB")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")

if __name__ == "__main__":
    print("🔍 CHESS MODEL VERIFICATION")
    print("=" * 60)
    
    # Analyze model
    analyze_model_info()
    
    # Test on dataset
    test_model_on_dataset()
    
    # Test on synthetic square
    test_model_on_single_square()
    
    print(f"\n🎉 All tests completed!")
    print(f"Check model_test_results/ folder for visual results")