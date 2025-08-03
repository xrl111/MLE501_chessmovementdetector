#!/usr/bin/env python3
"""
Clean project script - organize files for GitHub
"""
import os
import shutil
import glob

def clean_project():
    """Clean and organize project structure"""
    
    print("🧹 CLEANING PROJECT FOR GITHUB")
    print("=" * 50)
    
    # Files to keep (core functionality)
    keep_files = {
        # Core detection scripts
        'test_video_improved_moves.py',
        'test_video_flexible.py', 
        'test_any_chess_image.py',
        'test_model_simple.py',
        
        # Core detection (legacy but important)
        'chess_movement_detector_yolo.py',
        'chess_movement_detector.py',
        
        # Model and testing
        'piece model_tester.py',
        'convert_model_to_onnx.py',
        
        # Project files
        'README.md',
        'requirements.txt',
        '.gitignore',
        'clean_project.py'
    }
    
    # Files to remove (temporary/outdated)
    remove_patterns = [
        '*.mp4', '*.avi', '*.mov', '*.mkv',  # Video outputs
        '*.jpg', '*.jpeg', '*.png',          # Image outputs (except examples)
        'game_data_*.json',                  # Temporary data
        'flexible_game_data_*.json',         # Temporary data
        'detection_result_*.jpg',            # Result images
        'enhanced_*.mp4',                    # Processed videos
        'moves_*.mp4',                       # Move videos
        'detected_*.mp4',                    # Detection videos
        'single_test_result.jpg',            # Test results
        'frame_*.jpg',                       # Frame captures
        'test_video_adjustable.py',          # Outdated scripts
        'test_video_with_moves.py',          # Outdated scripts
        'test_video_detection.py',           # Outdated scripts
        'test_single_image.py'               # Outdated scripts
    ]
    
    # Create directories if needed
    os.makedirs('examples', exist_ok=True)
    os.makedirs('docs', exist_ok=True)
    
    # Remove temporary files
    removed_count = 0
    for pattern in remove_patterns:
        for file_path in glob.glob(pattern):
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"🗑️ Removed: {file_path}")
                    removed_count += 1
            except Exception as e:
                print(f"❌ Failed to remove {file_path}: {e}")
    
    # Move model validation to examples
    if os.path.exists('test_model_simple.py'):
        try:
            shutil.move('test_model_simple.py', 'examples/model_validation.py')
            print(f"📁 Moved: test_model_simple.py → examples/model_validation.py")
        except Exception as e:
            print(f"❌ Failed to move model validation: {e}")
    
    # List remaining Python files
    python_files = glob.glob('*.py')
    print(f"\n📄 REMAINING PYTHON FILES:")
    print("-" * 30)
    for py_file in sorted(python_files):
        size_kb = os.path.getsize(py_file) / 1024
        print(f"   ✅ {py_file:<30} ({size_kb:.1f} KB)")
    
    # Show directory structure
    print(f"\n📁 PROJECT STRUCTURE:")
    print("-" * 30)
    for root, dirs, files in os.walk('.'):
        # Skip hidden and large directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'yolo_dataset', 'chess_training', 'debug_frames']]
        
        level = root.replace('.', '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Limit to first 5 files per directory
            if not file.startswith('.'):
                print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")
    
    print(f"\n✅ CLEANUP SUMMARY:")
    print(f"   🗑️ Removed {removed_count} temporary files")
    print(f"   📦 Project ready for GitHub")
    print(f"   📄 Main scripts: {len(python_files)} Python files")
    
    # Check .gitignore
    if os.path.exists('.gitignore'):
        print(f"   ✅ .gitignore configured")
    else:
        print(f"   ⚠️ .gitignore missing")
    
    # Check requirements.txt
    if os.path.exists('requirements.txt'):
        print(f"   ✅ requirements.txt configured")
    else:
        print(f"   ⚠️ requirements.txt missing")
    
    print(f"\n🚀 Ready to push to GitHub!")
    print(f"📋 Next steps:")
    print(f"   1. git add .")
    print(f"   2. git commit -m 'Clean and organize project'")
    print(f"   3. git push origin main")

if __name__ == "__main__":
    clean_project()