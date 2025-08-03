#!/usr/bin/env python3
"""
Improved chess move detection with better logic and chess rules validation
"""
import cv2
import os
import sys
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
from collections import defaultdict, deque
import time
import json
import numpy as np
import math

def select_video_file():
    """Open a file dialog to select a video file."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Chọn video bàn cờ để test",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
            ("All files", "*.*")
        ]
    )
    return file_path

def get_board_bounds(detections, class_names):
    """Find board boundaries from detections"""
    board_boxes = []
    for j, (conf, cls, box) in enumerate(zip(
        detections.conf.cpu().numpy(), 
        detections.cls.cpu().numpy(),
        detections.xyxy.cpu().numpy()
    )):
        class_name = class_names[int(cls)] if int(cls) in class_names else 'unknown'
        if class_name in ['board', 'chessboards-and-pieces'] and conf > 0.7:
            board_boxes.append(box)
    
    if board_boxes:
        # Use the largest/most confident board detection
        largest_board = max(board_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
        return largest_board
    return None

def box_to_chess_position_improved(box, board_bounds=None, frame_width=None, frame_height=None):
    """Improved chess position mapping using board boundaries"""
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Use board bounds if available, otherwise use frame bounds
    if board_bounds is not None:
        board_x1, board_y1, board_x2, board_y2 = board_bounds
        # Normalize relative to board
        norm_x = (center_x - board_x1) / (board_x2 - board_x1)
        norm_y = (center_y - board_y1) / (board_y2 - board_y1)
    else:
        # Fallback to frame bounds
        norm_x = center_x / frame_width
        norm_y = center_y / frame_height
    
    # Clamp to valid range
    norm_x = max(0, min(1, norm_x))
    norm_y = max(0, min(1, norm_y))
    
    # Convert to chess coordinates
    file_idx = int(norm_x * 8)
    file_idx = max(0, min(7, file_idx))
    file_letter = chr(ord('a') + file_idx)
    
    rank_idx = int(norm_y * 8)
    rank_idx = max(0, min(7, rank_idx))
    rank_number = 8 - rank_idx
    
    return f"{file_letter}{rank_number}"

def is_valid_chess_move(piece, from_pos, to_pos):
    """Basic chess rules validation"""
    if from_pos == to_pos:
        return False
    
    from_file, from_rank = from_pos[0], int(from_pos[1])
    to_file, to_rank = to_pos[0], int(to_pos[1])
    
    file_diff = abs(ord(to_file) - ord(from_file))
    rank_diff = abs(to_rank - from_rank)
    
    piece_type = piece.lower()
    
    # Pawn moves
    if piece_type == 'p':
        # Forward only (simplified - not considering captures)
        if file_diff == 0:  # Same file
            if piece.isupper():  # White pawn
                return to_rank > from_rank and rank_diff <= 2
            else:  # Black pawn
                return to_rank < from_rank and rank_diff <= 2
        elif file_diff == 1 and rank_diff == 1:  # Diagonal capture
            return True
        return False
    
    # Rook moves
    elif piece_type == 'r':
        return file_diff == 0 or rank_diff == 0
    
    # Knight moves
    elif piece_type == 'n':
        return (file_diff == 2 and rank_diff == 1) or (file_diff == 1 and rank_diff == 2)
    
    # Bishop moves
    elif piece_type == 'b':
        return file_diff == rank_diff
    
    # Queen moves
    elif piece_type == 'q':
        return file_diff == 0 or rank_diff == 0 or file_diff == rank_diff
    
    # King moves
    elif piece_type == 'k':
        return file_diff <= 1 and rank_diff <= 1
    
    return True  # Allow unknown pieces

def calculate_move_distance(from_pos, to_pos):
    """Calculate euclidean distance between chess positions"""
    from_file, from_rank = ord(from_pos[0]) - ord('a'), int(from_pos[1]) - 1
    to_file, to_rank = ord(to_pos[0]) - ord('a'), int(to_pos[1]) - 1
    
    return math.sqrt((to_file - from_file)**2 + (to_rank - from_rank)**2)

def detect_pieces_in_frame_improved(model, frame, conf_threshold=0.6):
    """Enhanced piece detection with board-relative positioning"""
    try:
        results = model.predict(frame, verbose=False, conf=conf_threshold, save=False)
        
        if results and len(results) > 0:
            detections = results[0].boxes
            if detections is not None and len(detections) > 0:
                class_names = model.names
                
                # First find board boundaries
                board_bounds = get_board_bounds(detections, class_names)
                frame_height, frame_width = frame.shape[:2]
                
                piece_positions = {}
                detection_details = []
                
                for j, (conf, cls, box) in enumerate(zip(
                    detections.conf.cpu().numpy(), 
                    detections.cls.cpu().numpy(),
                    detections.xyxy.cpu().numpy()
                )):
                    class_name = class_names[int(cls)] if int(cls) in class_names else 'unknown'
                    
                    # Skip board detection for position tracking
                    if class_name in ['board', 'chessboards-and-pieces']:
                        continue
                        
                    if conf > 0.7:  # High confidence pieces only
                        x1, y1, x2, y2 = box.astype(int)
                        chess_pos = box_to_chess_position_improved(
                            box, board_bounds, frame_width, frame_height
                        )
                        
                        # Store best detection per position (highest confidence)
                        if chess_pos not in piece_positions or piece_positions[chess_pos]['confidence'] < conf:
                            piece_positions[chess_pos] = {
                                'piece': class_name,
                                'confidence': conf,
                                'box': (x1, y1, x2, y2)
                            }
                        
                        detection_details.append({
                            'class': class_name,
                            'conf': conf,
                            'box': (x1, y1, x2, y2),
                            'position': chess_pos
                        })
                
                return piece_positions, detection_details, results[0], board_bounds
        
        return {}, [], None, None
        
    except Exception as e:
        print(f"❌ Detection error: {e}")
        return {}, [], None, None

class MoveFilter:
    """Filter and validate chess moves"""
    
    def __init__(self, min_confidence=0.8, min_distance=0.5, stability_frames=3):
        self.min_confidence = min_confidence
        self.min_distance = min_distance
        self.stability_frames = stability_frames
        self.position_history = deque(maxlen=10)
        self.pending_moves = {}
        
    def add_frame_positions(self, positions):
        """Add frame positions to history"""
        self.position_history.append(positions)
        
    def detect_stable_moves(self):
        """Detect moves that are stable across multiple frames"""
        if len(self.position_history) < 2:
            return []
        
        prev_positions = self.position_history[-2]
        curr_positions = self.position_history[-1]
        
        validated_moves = []
        
        # Look for pieces that disappeared from old positions
        for prev_pos, prev_info in prev_positions.items():
            piece_type = prev_info['piece']
            prev_conf = prev_info['confidence']
            
            # Check if piece is no longer at this position
            if prev_pos not in curr_positions or curr_positions[prev_pos]['piece'] != piece_type:
                
                # Look for same piece type in new positions
                for curr_pos, curr_info in curr_positions.items():
                    if (curr_info['piece'] == piece_type and 
                        curr_pos != prev_pos and
                        curr_pos not in prev_positions):
                        
                        # Calculate move properties
                        distance = calculate_move_distance(prev_pos, curr_pos)
                        avg_confidence = (prev_conf + curr_info['confidence']) / 2
                        
                        # Apply filters
                        if (avg_confidence >= self.min_confidence and 
                            distance >= self.min_distance and
                            is_valid_chess_move(piece_type, prev_pos, curr_pos)):
                            
                            move_key = f"{piece_type}{prev_pos}-{curr_pos}"
                            
                            # Add to pending moves for stability check
                            if move_key not in self.pending_moves:
                                self.pending_moves[move_key] = {
                                    'move': move_key,
                                    'piece': piece_type,
                                    'from': prev_pos,
                                    'to': curr_pos,
                                    'confidence': avg_confidence,
                                    'distance': distance,
                                    'count': 1,
                                    'first_seen': len(self.position_history)
                                }
                            else:
                                self.pending_moves[move_key]['count'] += 1
        
        # Check for stable moves (seen multiple times)
        stable_moves = []
        for move_key, move_info in list(self.pending_moves.items()):
            frames_since_first = len(self.position_history) - move_info['first_seen']
            
            if move_info['count'] >= self.stability_frames or frames_since_first > 5:
                if move_info['count'] >= self.stability_frames:
                    stable_moves.append(move_info)
                # Remove from pending
                del self.pending_moves[move_key]
        
        return stable_moves

def annotate_frame_with_improved_moves(frame, detection_details, recent_moves, board_bounds=None):
    """Enhanced annotation with board highlighting"""
    annotated_frame = frame.copy()
    
    # Draw board boundary if detected
    if board_bounds is not None:
        x1, y1, x2, y2 = board_bounds.astype(int)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(annotated_frame, "BOARD", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Draw piece detections
    for detection in detection_details:
        class_name = detection['class']
        conf = detection['conf']
        x1, y1, x2, y2 = detection['box']
        position = detection['position']
        
        # Color coding
        if class_name.isupper():  # White pieces
            color = (255, 255, 255)
        elif class_name.islower():  # Black pieces  
            color = (0, 0, 0)
        else:
            color = (0, 255, 0)
        
        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # Position label
        label = f"{class_name}@{position}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 5), 
                     (x1 + label_size[0], y1), color, -1)
        text_color = (255, 255, 255) if class_name.islower() else (0, 0, 0)
        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
    
    # Enhanced moves panel
    frame_height, frame_width = frame.shape[:2]
    panel_width = 350
    panel_height = 250
    panel_x = frame_width - panel_width - 10
    panel_y = 10
    
    # Panel background
    overlay = annotated_frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), 
                  (panel_x + panel_width, panel_y + panel_height), 
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.8, annotated_frame, 0.2, 0, annotated_frame)
    
    # Panel border
    cv2.rectangle(annotated_frame, (panel_x, panel_y), 
                  (panel_x + panel_width, panel_y + panel_height), 
                  (255, 255, 255), 2)
    
    # Title
    cv2.putText(annotated_frame, "VALIDATED MOVES", 
                (panel_x + 10, panel_y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Recent moves with validation info
    y_offset = 50
    for i, move_info in enumerate(recent_moves):
        if i >= 5:
            break
            
        move_text = f"{i+1}. {move_info['move']}"
        conf_text = f"conf:{move_info['confidence']:.2f}"
        dist_text = f"dist:{move_info.get('distance', 0):.1f}"
        
        # Move notation
        cv2.putText(annotated_frame, move_text, 
                    (panel_x + 10, panel_y + y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Move details
        details = f"{conf_text} {dist_text}"
        cv2.putText(annotated_frame, details, 
                    (panel_x + 10, panel_y + y_offset + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        y_offset += 35
    
    # Statistics
    cv2.putText(annotated_frame, f"Pieces: {len(detection_details)}", 
                (panel_x + 10, panel_y + panel_height - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    cv2.putText(annotated_frame, f"Valid moves: {len(recent_moves)}", 
                (panel_x + 10, panel_y + panel_height - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return annotated_frame

def test_video_improved_moves(video_path, model_path="model_dataset/PieceDetection/best.pt"):
    """Test video with improved move detection logic"""
    
    print(f"♟️ ENHANCED CHESS MOVE DETECTION")
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
    
    print(f"📊 Processing with enhanced filters:")
    print(f"   🎯 Min confidence: 80%")
    print(f"   📏 Min move distance: 0.5 squares")
    print(f"   ✅ Chess rules validation: ON")
    print(f"   🕒 Stability check: 3 frames")
    
    # Setup video writer
    output_path = f"enhanced_{os.path.splitext(os.path.basename(video_path))[0]}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize move filter
    move_filter = MoveFilter(min_confidence=0.8, min_distance=0.5, stability_frames=3)
    
    # Tracking variables
    frame_idx = 0
    validated_moves = []
    recent_moves = deque(maxlen=5)
    
    print(f"\n🔄 Processing with enhanced move detection...")
    print("=" * 70)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            current_time = frame_idx / fps if fps > 0 else frame_idx
            
            # Detect pieces with enhanced logic
            positions, detection_details, results, board_bounds = detect_pieces_in_frame_improved(model, frame)
            
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
                
                print(f"✅ VALID MOVE: {move['move']} at {current_time:.1f}s")
                print(f"   📊 Confidence: {move['confidence']:.2f}, Distance: {move['distance']:.1f}")
            
            # Annotate frame
            annotated_frame = annotate_frame_with_improved_moves(
                frame, detection_details, list(recent_moves), board_bounds
            )
            
            # Add frame info
            info_text = f"Frame: {frame_idx} | Time: {current_time:.1f}s | Valid moves: {len(validated_moves)}"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display and save
            cv2.imshow('Enhanced Chess Detection', annotated_frame)
            out.write(annotated_frame)
            
            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                cv2.waitKey(0)
    
    except KeyboardInterrupt:
        print(f"\n⏹️ Interrupted by user")
    
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    # Final summary
    print(f"\n📈 ENHANCED DETECTION SUMMARY:")
    print("=" * 70)
    print(f"🎞️ Processed frames: {frame_idx}")
    print(f"✅ Validated moves: {len(validated_moves)}")
    
    if validated_moves:
        print(f"\n🎯 VALIDATED MOVES:")
        print("-" * 50)
        for i, move in enumerate(validated_moves):
            print(f"{i+1:2d}. {move['move']:<12} at {move['time']:6.1f}s")
            print(f"     conf:{move['confidence']:.2f} dist:{move['distance']:.1f}")
    else:
        print(f"ℹ️  No valid moves detected (high filter standards)")
    
    print(f"\n💾 Enhanced video saved: {output_path}")
    return True

def main():
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = select_video_file()
        if not video_path:
            return
    
    test_video_improved_moves(video_path)

if __name__ == "__main__":
    main()