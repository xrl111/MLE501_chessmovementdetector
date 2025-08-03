#!/usr/bin/env python3
"""
Flexible chess move detection with adjustable sensitivity
"""
import sys
from test_video_improved_moves import *

def test_video_flexible_params(video_path, model_path="model_dataset/PieceDetection/best.pt"):
    """Test video with more flexible parameters for real games"""
    
    print(f"♟️ FLEXIBLE CHESS MOVE DETECTION")
    print("=" * 70)
    print(f"📹 Video: {os.path.basename(video_path)}")
    
    # Load model
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    try:
        model = YOLO(model_path)
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video file")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📊 Flexible parameters:")
    print(f"   🎯 Min confidence: 70% (reduced from 80%)")
    print(f"   📏 Min move distance: 0.3 squares (reduced from 0.5)")
    print(f"   ✅ Chess rules validation: ON")
    print(f"   🕒 Stability check: 2 frames (reduced from 3)")
    
    # Setup video writer
    output_path = f"flexible_{os.path.splitext(os.path.basename(video_path))[0]}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize move filter with more flexible parameters
    move_filter = MoveFilter(min_confidence=0.7, min_distance=0.3, stability_frames=2)
    
    # Tracking variables
    frame_idx = 0
    validated_moves = []
    recent_moves = deque(maxlen=5)
    
    print(f"\n🔄 Processing with flexible detection...")
    print("=" * 70)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            current_time = frame_idx / fps if fps > 0 else frame_idx
            
            # Detect pieces with enhanced logic
            positions, detection_details, results, board_bounds = detect_pieces_in_frame_improved(model, frame, conf_threshold=0.5)
            
            # Add to move filter
            move_filter.add_frame_positions(positions)
            
            # Get validated moves
            stable_moves = move_filter.detect_stable_moves()
            
            # Process new validated moves
            for move in stable_moves:
                move_info = {
                    'frame': frame_idx,
                    'time': current_time,
                    **move
                }
                validated_moves.append(move_info)
                recent_moves.append(move_info)
                
                print(f"✅ MOVE: {move['move']} at {current_time:.1f}s")
                print(f"   📊 Confidence: {move['confidence']:.2f}, Distance: {move['distance']:.1f}")
            
            # Annotate frame
            annotated_frame = annotate_frame_with_improved_moves(
                frame, detection_details, list(recent_moves), board_bounds
            )
            
            # Add frame info
            info_text = f"Frame: {frame_idx} | Time: {current_time:.1f}s | Moves: {len(validated_moves)}"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display and save
            cv2.imshow('Flexible Chess Detection', annotated_frame)
            out.write(annotated_frame)
            
            # Show progress every 100 frames
            if frame_idx % 100 == 0:
                progress = (frame_idx / frame_count) * 100
                print(f"📈 Progress: {progress:.1f}% | Current moves: {len(validated_moves)}")
            
            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(f"\n⏹️ Stopped by user")
                break
            elif key == ord(' '):
                print(f"⏸️ Paused - Press any key to continue...")
                cv2.waitKey(0)
            elif key == ord('m'):
                print(f"\n📋 CURRENT MOVES:")
                for i, move in enumerate(validated_moves[-5:]):
                    print(f"   {i+1}. {move['move']} at {move['time']:.1f}s")
    
    except KeyboardInterrupt:
        print(f"\n⏹️ Interrupted by user")
    
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    # Save game data
    if validated_moves:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        json_path = f"flexible_game_data_{base_name}.json"
        
        game_data = {
            'parameters': {
                'min_confidence': 0.7,
                'min_distance': 0.3,
                'stability_frames': 2,
                'chess_rules': True
            },
            'moves': [
                {
                    'frame': move_info['frame'],
                    'time': move_info['time'],
                    'move': move_info['move'],
                    'piece': move_info['piece'],
                    'from': move_info['from'],
                    'to': move_info['to'],
                    'confidence': float(move_info['confidence']),
                    'distance': float(move_info.get('distance', 0))
                }
                for move_info in validated_moves
            ],
            'total_moves': len(validated_moves),
            'game_summary': {
                'duration_frames': frame_idx,
                'moves_detected': len(validated_moves),
                'avg_moves_per_minute': len(validated_moves) / (frame_idx / fps / 60) if frame_idx > 0 else 0
            }
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(game_data, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Game data saved: {json_path}")
    
    # Final summary
    print(f"\n📈 FLEXIBLE DETECTION SUMMARY:")
    print("=" * 70)
    print(f"🎞️ Processed frames: {frame_idx}")
    print(f"✅ Validated moves: {len(validated_moves)}")
    print(f"⏱️ Game duration: {frame_idx/fps:.1f} seconds")
    
    if validated_moves:
        print(f"\n🎯 DETECTED MOVES:")
        print("-" * 60)
        for i, move in enumerate(validated_moves):
            print(f"{i+1:2d}. {move['move']:<12} at {move['time']:6.1f}s (conf:{move['confidence']:.2f} dist:{move['distance']:.1f})")
    else:
        print(f"ℹ️  No valid moves detected - try with a video containing actual chess moves")
    
    print(f"\n💾 Output files:")
    print(f"   🎬 Video: {output_path}")
    if validated_moves:
        print(f"   📄 Data: flexible_game_data_{os.path.splitext(os.path.basename(video_path))[0]}.json")
    
    return True

def main():
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = select_video_file()
        if not video_path:
            return
    
    test_video_flexible_params(video_path)

if __name__ == "__main__":
    main()