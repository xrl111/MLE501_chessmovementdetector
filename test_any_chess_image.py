#!/usr/bin/env python3
"""
Interactive script to test chess detection on any image
"""
import cv2
import os
import sys
from ultralytics import YOLO
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

def select_image_file():
    """Open a file dialog to select an image file."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh bàn cờ để test",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("All files", "*.*")
        ]
    )
    return file_path

def test_chess_image(image_path, model_path="model_dataset/PieceDetection/best.pt"):
    """Test chess piece detection on any image"""
    
    print(f"🔍 TESTING CHESS PIECE DETECTION")
    print("=" * 60)
    print(f"📷 Image: {os.path.basename(image_path)}")
    
    # Load model
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    try:
        model = YOLO(model_path)
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Load image
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
        
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Failed to load image")
        return
        
    print(f"📏 Image dimensions: {img.shape[1]}x{img.shape[0]} pixels")
    
    # Run chess piece detection
    try:
        print(f"🔄 Running detection...")
        results = model.predict(img, verbose=False, conf=0.4, save=False)
        
        if results and len(results) > 0:
            detections = results[0].boxes
            if detections is not None and len(detections) > 0:
                print(f"🎯 Found {len(detections)} detections")
                
                # Get class names from model
                class_names = model.names
                
                # Count pieces by type
                piece_counts = {}
                high_conf_pieces = {}
                
                print(f"\n📋 Detection Details:")
                print("-" * 40)
                
                for j, (conf, cls, box) in enumerate(zip(
                    detections.conf.cpu().numpy(), 
                    detections.cls.cpu().numpy(),
                    detections.xyxy.cpu().numpy()
                )):
                    class_name = class_names[int(cls)] if int(cls) in class_names else 'unknown'
                    
                    # Count all pieces
                    piece_counts[class_name] = piece_counts.get(class_name, 0) + 1
                    
                    # Count high confidence pieces (>0.7)
                    if conf > 0.7:
                        high_conf_pieces[class_name] = high_conf_pieces.get(class_name, 0) + 1
                        x1, y1, x2, y2 = box.astype(int)
                        print(f"   {j+1:2d}. {class_name:12s} conf:{conf:.3f} at [{x1:3d},{y1:3d}] to [{x2:3d},{y2:3d}]")
                
                # Summary
                print(f"\n📊 DETECTION SUMMARY:")
                print("-" * 40)
                print(f"{'Piece Type':<12} {'Total':<8} {'High Conf':<10} {'Expected':<10}")
                print("-" * 40)
                
                expected_pieces = {
                    'P': 8, 'p': 8,  # Pawns
                    'R': 2, 'r': 2,  # Rooks  
                    'N': 2, 'n': 2,  # Knights
                    'B': 2, 'b': 2,  # Bishops
                    'Q': 1, 'q': 1,  # Queens
                    'K': 1, 'k': 1,  # Kings
                    'board': 1,      # Board detection
                    'chessboards-and-pieces': '?'  # Special class
                }
                
                total_pieces = 0
                for piece_type in ['P', 'R', 'N', 'B', 'Q', 'K', 'p', 'r', 'n', 'b', 'q', 'k', 'board']:
                    total = piece_counts.get(piece_type, 0)
                    high_conf = high_conf_pieces.get(piece_type, 0)
                    expected = expected_pieces.get(piece_type, 0)
                    
                    if total > 0 or piece_type in ['P', 'p', 'R', 'r', 'N', 'n', 'B', 'b', 'Q', 'q', 'K', 'k']:
                        print(f"{piece_type:<12} {total:<8} {high_conf:<10} {expected:<10}")
                        if piece_type != 'board':
                            total_pieces += high_conf
                
                print("-" * 40)
                print(f"{'TOTAL':<12} {sum(piece_counts.values()):<8} {total_pieces:<10} {'32':<10}")
                
                # Save result image with detections
                result_img = results[0].plot()
                result_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
                
                # Create output filename
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_path = f"detection_result_{base_name}.jpg"
                cv2.imwrite(output_path, result_img_bgr)
                
                print(f"\n💾 Result image saved: {output_path}")
                print(f"🖼️  You can view the annotated image to see bounding boxes")
                
                return True
                
            else:
                print(f"⚪ No chess pieces detected in the image")
                print(f"💡 Try with:")
                print(f"   - Better lit chess board image")
                print(f"   - Image with clearer piece shapes")
                print(f"   - Different camera angle")
        else:
            print(f"❌ No detection results from model")
            
    except Exception as e:
        print(f"❌ Detection failed: {e}")
        return False
    
    return False

def main():
    """Main function"""
    print("🏁 CHESS PIECE DETECTION TESTER")
    print("=" * 60)
    
    # Check if image path provided as argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Let user select image
        print("📁 Please select a chess board image to test...")
        image_path = select_image_file()
        
        if not image_path:
            print("❌ No image selected. Exiting.")
            return
    
    # Test the image
    success = test_chess_image(image_path)
    
    if success:
        print(f"\n🎉 Detection completed successfully!")
        print(f"🔍 The model found chess pieces in your image")
    else:
        print(f"\n⚠️  Detection completed but with issues")
        print(f"💭 Consider trying a different image or adjusting lighting")

if __name__ == "__main__":
    main()