"""
Convert trained PyTorch model to ONNX format (Windows compatible)
"""
import torch
import os
import shutil
from pathlib import Path

def convert_pytorch_to_onnx():
    """Convert the trained model to ONNX format without dependencies issues"""
    
    model_path = "chess_training_windows/balanced_yolov8s/weights/best.pt"
    output_path = "model_dataset/PieceDetection/best.onnx"
    
    print("🔄 Converting PyTorch model to ONNX...")
    print(f"📁 Input: {model_path}")
    print(f"📁 Output: {output_path}")
    
    if not os.path.exists(model_path):
        print("❌ Trained model not found!")
        return False
    
    try:
        # Method 1: Use ultralytics with simplified export
        from ultralytics import YOLO
        
        print("📦 Loading trained model...")
        model = YOLO(model_path)
        
        print("🔧 Exporting to ONNX (simplified)...")
        # Use simplified export without problematic dependencies
        export_path = model.export(
            format='onnx',
            dynamic=False,
            simplify=False,  # Disable simplify to avoid onnx dependency
            opset=11,        # Use older opset for compatibility
            imgsz=640
        )
        
        # Move to correct location
        os.makedirs("model_dataset/PieceDetection", exist_ok=True)
        shutil.copy2(export_path, output_path)
        
        print(f"✅ ONNX model created: {output_path}")
        print(f"📏 File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        
        # Method 2: Manual PyTorch export
        try:
            print("🔄 Trying manual PyTorch export...")
            
            # Load model
            checkpoint = torch.load(model_path, map_location='cpu')
            model = checkpoint['model'].float()
            model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(1, 3, 640, 640)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['images'],
                output_names=['output'],
                dynamic_axes={
                    'images': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            print(f"✅ Manual ONNX export successful: {output_path}")
            return True
            
        except Exception as e2:
            print(f"❌ Manual export also failed: {e2}")
            
            # Method 3: Copy PyTorch model and modify detection script
            print("🔄 Plan C: Copy PyTorch model for direct use...")
            
            try:
                pt_output = "model_dataset/PieceDetection/best.pt"
                shutil.copy2(model_path, pt_output)
                print(f"✅ PyTorch model copied: {pt_output}")
                print("💡 Will modify detection script to use .pt instead of .onnx")
                return "pytorch"
                
            except Exception as e3:
                print(f"❌ All methods failed: {e3}")
                return False

def modify_detection_for_pytorch():
    """Modify detection script to use PyTorch model directly"""
    
    detection_file = "chess_movement_detector_yolo.py"
    
    print("🔧 Modifying detection script for PyTorch model...")
    
    try:
        with open(detection_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace ONNX loading with PyTorch loading
        old_load = '''def load_model(model_path="model_dataset/PieceDetection/best.onnx"):
    """Load ONNX model for piece classification"""
    if not os.path.exists(model_path):
        logging.warning(f"ONNX model not found at {model_path}. Running in demo mode without piece classification.")
        return None
    
    session = ort.InferenceSession(model_path)
    logging.info(f"Loaded ONNX model from {model_path}")
    return session'''
    
        new_load = '''def load_model(model_path="model_dataset/PieceDetection/best.pt"):
    """Load PyTorch model for piece classification"""
    # Try PyTorch model first
    pt_path = model_path.replace('.onnx', '.pt')
    if os.path.exists(pt_path):
        try:
            from ultralytics import YOLO
            model = YOLO(pt_path)
            logging.info(f"Loaded PyTorch model from {pt_path}")
            return model
        except Exception as e:
            logging.warning(f"Failed to load PyTorch model: {e}")
    
    # Fallback to ONNX
    if os.path.exists(model_path):
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(model_path)
            logging.info(f"Loaded ONNX model from {model_path}")
            return session
        except Exception as e:
            logging.warning(f"Failed to load ONNX model: {e}")
    
    logging.warning(f"No model found. Running in demo mode without piece classification.")
    return None'''
        
        content = content.replace(old_load, new_load)
        
        # Update inference function
        old_inference = '''def match_piece(square_img, img_name, model_session, class_names, frame_idx, threshold=0.8, debug=False):
    """Classify chess piece in a square image using ONNX model"""
    
    if model_session is None:
        # Demo mode - alternating pieces
        row, col = int(img_name.split('_')[1][1:]), int(img_name.split('_')[2][1:].split('.')[0])
        if (row + col) % 2 == 0:
            return 'empty'
        else:
            demo_pieces = ['P', 'p', 'R', 'r', 'N', 'n', 'B', 'b', 'Q', 'q', 'K', 'k']
            return demo_pieces[(row + col) % len(demo_pieces)]
    
    try:'''
        
        new_inference = '''def match_piece(square_img, img_name, model_session, class_names, frame_idx, threshold=0.8, debug=False):
    """Classify chess piece in a square image using PyTorch or ONNX model"""
    
    if model_session is None:
        # Demo mode - alternating pieces
        row, col = int(img_name.split('_')[1][1:]), int(img_name.split('_')[2][1:].split('.')[0])
        if (row + col) % 2 == 0:
            return 'empty'
        else:
            demo_pieces = ['P', 'p', 'R', 'r', 'N', 'n', 'B', 'b', 'Q', 'q', 'K', 'k']
            return demo_pieces[(row + col) % len(demo_pieces)]
    
    # Check if it's PyTorch model
    try:
        if hasattr(model_session, 'predict'):  # Ultralytics YOLO model
            # Use PyTorch model
            results = model_session.predict(square_img, verbose=False)
            if results and len(results) > 0 and len(results[0].boxes) > 0:
                # Get highest confidence detection
                confidences = results[0].boxes.conf.cpu().numpy()
                best_idx = confidences.argmax()
                if confidences[best_idx] > threshold:
                    class_id = int(results[0].boxes.cls[best_idx].cpu().numpy())
                    return class_names[class_id] if class_id < len(class_names) else 'empty'
            return 'empty'
        else:
            # Use ONNX model (original code)'''
            
        content = content.replace(old_inference, new_inference)
        
        with open(detection_file, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print(f"✅ Detection script updated for PyTorch model")
        return True
        
    except Exception as e:
        print(f"❌ Failed to modify detection script: {e}")
        return False

if __name__ == "__main__":
    print("🔄 Converting trained model for deployment...")
    
    result = convert_pytorch_to_onnx()
    
    if result == "pytorch":
        # ONNX failed, use PyTorch directly
        modify_detection_for_pytorch()
        print("\n🎉 Model ready! Using PyTorch format.")
        print("📝 Detection script updated to use .pt model")
        
    elif result:
        print("\n🎉 ONNX conversion successful!")
        print("📝 Model ready for detection")
        
    else:
        print("\n❌ Model conversion failed")
        print("💡 Manual solution needed")
    
    print(f"\n🧪 Test with: python chess_movement_detector_yolo.py")