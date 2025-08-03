"""
Update hardcoded paths in chess movement detector for Windows
"""
import os
import re

def update_file_paths(file_path, path_mappings):
    """Update hardcoded paths in a Python file"""
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    
    print(f"Updating {file_path}...")
    
    # Read file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Apply path mappings
    for old_path, new_path in path_mappings.items():
        content = content.replace(old_path, new_path)
    
    # Update any remaining macOS/Linux paths
    macos_patterns = [
        (r'/Users/[^/]+/[^"\']+', 'model_dataset/PieceDetection/'),
        (r'/content/[^"\']+', './'),
        (r'~/[^"\']+', './'),
    ]
    
    for pattern, replacement in macos_patterns:
        content = re.sub(pattern, replacement, content)
    
    # Write back if changed
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Updated {file_path}")
        return True
    else:
        print(f"✓ {file_path} already up to date")
        return False

def main():
    """Update all files for Windows"""
    print("=== Updating Chess Movement Detector for Windows ===\n")
    
    # Path mappings for common hardcoded paths
    path_mappings = {
        # macOS paths to relative Windows paths
        "/Users/macintoshhd/Downloads/MSE24/2025_semester_02/MLE501.9/final/model_dataset/PieceDetection/best.onnx": 
            "model_dataset/PieceDetection/best.onnx",
        "/Users/macintoshhd/Downloads/MSE24/2025_semester_02/MLE501.9/final/model_dataset/PieceDetection/piece_classifier_model.h5": 
            "model_dataset/PieceDetection/piece_classifier_model.h5",
        "/Users/macintoshhd/Downloads/MSE24/2025_semester_02/MLE501.9/final/templates": 
            "templates",
        "/content/chess_dataset/": 
            "./yolo_dataset/",
        "/Users/macintoshhd/Downloads/MSE24/2025_semester_02/MLE501.9/final/model_dataset/PieceDetection/dataset": 
            "./yolo_dataset/train",
    }
    
    # Files to update
    files_to_update = [
        "chess_movement_detector_yolo.py",
        "chess_movement_detector.py", 
        "piece model_tester.py",
        "model_dataset/PieceDetection/PiecDetectionTrainer.py",
        "model_dataset/PieceDetection/PieceDetectorModel_Yolo_collab.py"
    ]
    
    updated_files = []
    
    for file_path in files_to_update:
        if update_file_paths(file_path, path_mappings):
            updated_files.append(file_path)
    
    print(f"\n=== Update Summary ===")
    if updated_files:
        print(f"Updated {len(updated_files)} files:")
        for file_path in updated_files:
            print(f"  ✅ {file_path}")
    else:
        print("✓ All files already up to date")
    
    print("\n📝 Additional manual updates needed:")
    print("1. Update any remaining hardcoded paths in custom scripts")
    print("2. Ensure model files exist in model_dataset/PieceDetection/")
    print("3. Test with: python chess_movement_detector_yolo.py")

if __name__ == "__main__":
    main()