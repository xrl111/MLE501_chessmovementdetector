import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import tkinter as tk
from tkinter import filedialog
from copy import deepcopy
import onnxruntime as ort

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Match result
match_result = []

# Erosion kernel
EROSION_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Global variables for manual point selection
selected_points = []
click_count = 0

def select_video_file():
    """Open a file dialog to select a video file."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video files", "*.mp4 *.mkv *.avi")]
    )
    return file_path

def get_video_duration(video_path):
    """Get the duration of a video in seconds."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video file: {video_path}")
        raise ValueError("Cannot open video file")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    
    if fps <= 0:
        logging.error("Invalid FPS value in video")
        raise ValueError("Invalid FPS value")
    
    duration = frame_count / fps
    logging.info(f"Video duration: {duration:.2f} seconds")
    return duration

def extract_digital_board(image, debug=False):
    """Detect chessboard region from image."""
    print('Detecting first chessboard square from image...')
    h, w = image.shape[:2]
    crop = image[0:h, 0:w]

    if crop.size == 0:
        raise ValueError("Cropped image is empty")

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 3)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(closed, 50, 150)

    contour_modes = [cv2.RETR_EXTERNAL, cv2.RETR_TREE]
    best_quad = None

    if debug:
        os.makedirs('./debug_frames', exist_ok=True)
        cv2.imwrite('./debug_frames/_1_gray.png', gray)
        cv2.imwrite('./debug_frames/_2_blurred.png', blurred)
        cv2.imwrite('./debug_frames/_3_thresh.png', thresh)
        cv2.imwrite('./debug_frames/_4_closed.png', closed)
        cv2.imwrite('./debug_frames/_5_edges.png', edges)

    for mode in contour_modes:
        contours, _ = cv2.findContours(edges.copy(), mode, cv2.CHAIN_APPROX_SIMPLE)

        if debug:
            debug_img = crop.copy()
            cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)
            cv2.imwrite(f'./debug_frames/_contours_mode_{mode}.png', debug_img)

        for c in sorted(contours, key=cv2.contourArea, reverse=True):
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)

            if len(approx) == 4 and cv2.contourArea(approx) > 5000 and cv2.isContourConvex(approx):
                pts = approx.reshape(4, 2)
                rect = order_points(pts)

                side_lengths = [
                    np.linalg.norm(rect[0] - rect[1]),
                    np.linalg.norm(rect[1] - rect[2]),
                    np.linalg.norm(rect[2] - rect[3]),
                    np.linalg.norm(rect[3] - rect[0])
                ]

                if min(side_lengths) < 50:
                    continue

                aspect_ratio = max(side_lengths) / min(side_lengths)
                if 0.8 < aspect_ratio < 1.2:
                    best_quad = rect
                    break
        if best_quad is not None:
            break

    if best_quad is None:
        print("No valid square detected.")
        return crop, None

    square_w, square_h = 75, 75
    board_w, board_h = 8 * square_w, 8 * square_h

    top_left = best_quad[0]

    chessboard_corners = np.array([
        top_left,
        top_left + np.array([board_w, 0]),
        top_left + np.array([board_w, board_h]),
        top_left + np.array([0, board_h])
    ], dtype=np.float32)

    if debug:
        debug_img = crop.copy()
        cv2.polylines(debug_img, [best_quad.astype(int)], True, (255, 0, 0), 3)
        cv2.polylines(debug_img, [chessboard_corners.astype(int)], True, (0, 0, 255), 3)
        cv2.imwrite('./debug_frames/_detected_squares.png', debug_img)

    print(f"First square corners: {best_quad}")
    print(f"Computed chessboard corners: {chessboard_corners}")

    return crop, chessboard_corners

def order_points(pts):
    """Return consistent order: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def mouse_callback(event, x, y, flags, param):
    """Callback for mouse clicks to select chessboard corners."""
    global selected_points, click_count
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append([x, y])
        click_count += 1
        logging.info(f"Point {click_count} selected: ({x}, {y})")

def select_board_corners(frame):
    """Allow user to manually select four chessboard corners."""
    global selected_points, click_count
    selected_points = []
    click_count = 0
    
    window_name = "Select Chessboard Corners (Click 4 points, press 'q' to confirm)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    while click_count < 4:
        display_frame = frame.copy()
        for i, pt in enumerate(selected_points):
            cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
            cv2.putText(display_frame, f"P{i+1}", (int(pt[0]) + 10, int(pt[1])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow(window_name, display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') and click_count == 4:
            break
    
    cv2.destroyWindow(window_name)
    if len(selected_points) == 4:
        return np.array(selected_points, dtype="float32")
    else:
        logging.error("Exactly 4 points must be selected")
        raise ValueError("Exactly 4 points must be selected")

def blur_border(img, border_size=10, blur_kernel=(15, 15)):
    """Apply a blurred border to an image."""
    h, w = img.shape[:2]
    blurred = cv2.GaussianBlur(img, blur_kernel, 0)

    mask = np.zeros((h, w), dtype=np.float32)
    cv2.rectangle(mask, (border_size, border_size), (w - border_size, h - border_size), 1, -1)
    mask = cv2.GaussianBlur(mask, (border_size * 2 + 1, border_size * 2 + 1), 0)

    if len(img.shape) == 2:
        result = img.astype(np.float32) * mask + blurred.astype(np.float32) * (1 - mask)
    else:
        result = img.astype(np.float32)
        for c in range(3):
            result[..., c] = result[..., c] * mask + blurred[..., c] * (1 - mask)

    return np.clip(result, 0, 255).astype(np.uint8)

def load_model(model_path="model_dataset/PieceDetection/best.pt"):
    """Load the trained PyTorch model for chess piece classification."""
    # Try PyTorch model first (our trained model)
    pt_path = model_path.replace('.onnx', '.pt')
    if os.path.exists(pt_path):
        try:
            from ultralytics import YOLO
            model = YOLO(pt_path)
            logging.info(f"✅ Loaded trained PyTorch model from {pt_path}")
            return model
        except Exception as e:
            logging.warning(f"Failed to load PyTorch model: {e}")
    
    # Fallback to ONNX if available
    onnx_path = model_path.replace('.pt', '.onnx')
    if os.path.exists(onnx_path):
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(onnx_path)
            logging.info(f"Loaded ONNX model from {onnx_path}")
            return session
        except Exception as e:
            logging.warning(f"Failed to load ONNX model: {e}")
    
    logging.warning(f"No trained model found. Running in demo mode without piece classification.")
    return None

def load_class_names():
    """Return class names corresponding to model output indices."""
    class_names = ['B', 'K', 'N', 'P', 'Q', 'R', 'b', 'k', 'n', 'p', 'q', 'r', 'empty']
    return class_names

def match_piece(square_img, img_name, model_session, class_names, frame_idx, threshold=0.8, debug=False):
    """Classify a chess piece using either PyTorch YOLO or ONNX model."""
    if square_img is None or square_img.size == 0 or square_img.shape[0] == 0 or square_img.shape[1] == 0:
        logging.warning(f"Empty or invalid square image: {img_name}")
        return None

    # Demo mode: no model available
    if model_session is None:
        # Return dummy classification for testing (alternating pattern)
        row, col = int(img_name.split('_')[1][1:]), int(img_name.split('_')[2][1:].split('.')[0])
        if (row + col) % 2 == 0:
            return 'empty'  # Empty squares
        else:
            demo_pieces = ['P', 'p', 'R', 'r', 'N', 'n', 'B', 'b', 'Q', 'q', 'K', 'k']
            return demo_pieces[(row + col) % len(demo_pieces)]

    # Ensure square_img is 3-channel (BGR)
    if len(square_img.shape) == 2:
        logging.debug(f"Converting grayscale image {img_name} to BGR")
        square_img = cv2.cvtColor(square_img, cv2.COLOR_GRAY2BGR)
    elif len(square_img.shape) == 3 and square_img.shape[2] != 3:
        logging.error(f"Invalid image channels for {img_name}: {square_img.shape[2]}")
        return None

    try:
        # Check if it's a PyTorch YOLO model (our trained model)
        if hasattr(model_session, 'predict'):  # Ultralytics YOLO model
            # Use PyTorch YOLO model for object detection
            results = model_session.predict(square_img, verbose=False, conf=threshold)
            
            if results and len(results) > 0 and len(results[0].boxes) > 0:
                # Get highest confidence detection
                confidences = results[0].boxes.conf.cpu().numpy()
                best_idx = confidences.argmax()
                best_conf = confidences[best_idx]
                
                if best_conf >= threshold:
                    class_id = int(results[0].boxes.cls[best_idx].cpu().numpy())
                    if class_id < len(class_names):
                        best_match = class_names[class_id]
                        match_result.append(f"{img_name} match = {best_match} (conf: {best_conf:.2f})")
                        
                        if debug:
                            frame_debug_dir = f'./debug_frames/frame_{frame_idx:03d}/match'
                            os.makedirs(frame_debug_dir, exist_ok=True)
                            cv2.imwrite(f'{frame_debug_dir}/{img_name}_input.png', square_img)
                            with open(f'{frame_debug_dir}/{img_name}_detection.txt', 'w') as f:
                                f.write(f"Best detection: {best_match} (conf: {best_conf:.4f})\n")
                                f.write(f"All detections:\n")
                                for i, (cls_id, conf) in enumerate(zip(results[0].boxes.cls.cpu().numpy(), confidences)):
                                    cls_name = class_names[int(cls_id)] if int(cls_id) < len(class_names) else 'unknown'
                                    f.write(f"  {cls_name}: {conf:.4f}\n")
                        
                        return best_match
            
            # No confident detection found, assume empty
            return 'empty'
            
        else:
            # Use ONNX model (original classification logic)
            square_resized = cv2.cvtColor(square_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            square_resized = square_resized.astype(np.float32) / 255.0  # Normalize to [0,1]
            square_resized = square_resized.transpose(2, 0, 1)  # HWC to CHW
            img_array = np.expand_dims(square_resized, axis=0)  # Add batch dimension

            logging.debug(f"Input shape for {img_name}: {img_array.shape}")

            input_name = model_session.get_inputs()[0].name
            outputs = model_session.run(None, {input_name: img_array})
            probabilities = outputs[0][0]  # Assuming softmax output
            max_prob = np.max(probabilities)
            max_idx = np.argmax(probabilities)

            if max_prob >= threshold:
                best_match = class_names[max_idx]
                match_result.append(f"{img_name} match = {best_match} (prob: {max_prob:.2f})")

                if debug:
                    frame_debug_dir = f'./debug_frames/frame_{frame_idx:03d}/match'
                    os.makedirs(frame_debug_dir, exist_ok=True)
                    cv2.imwrite(f'{frame_debug_dir}/{img_name}_input.png', square_img)
                    with open(f'{frame_debug_dir}/{img_name}_probs.txt', 'w') as f:
                        for cls, prob in zip(class_names, probabilities):
                            f.write(f"{cls}: {prob:.4f}\n")

                return best_match
            return None

    except Exception as e:
        logging.error(f"Model inference error for {img_name}: {e}")
        return None

def warp_board(crop, points):
    """Warp the chessboard region to a standard 552x552 image."""
    rect = order_points(points)
    dst = np.array([
        [0, 0],
        [551, 0],
        [551, 551],
        [0, 551]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(crop, M, (552, 552))

    if not warped.size:
        raise ValueError("Warped image is empty or invalid")

    return warped, M

def split_into_squares(board_img, debug_dir="./debug_frames"):
    """Split the warped board into 8x8 squares and save debug images."""
    os.makedirs(debug_dir, exist_ok=True)
    squares = []
    square_names = []
    square_positions = []
    height, width = board_img.shape[:2]
    dy, dx = 69, 69

    if height < 8 * dy or width < 8 * dx:
        logging.warning(f"Warped board too small: {width}x{height}, need at least {8*dx}x{8*dy}")
        top_pad = max(0, 8 * dy - height)
        left_pad = max(0, 8 * dx - width)
        board_img = cv2.copyMakeBorder(board_img, 0, top_pad, 0, left_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        logging.info(f"Padded board to {board_img.shape[1]}x{board_img.shape[0]}")

    for row in range(8):
        for col in range(8):
            y_start, y_end = row * dy, (row + 1) * dy
            x_start, x_end = col * dx, (col + 1) * dx
            square = board_img[y_start:y_end, x_start:x_end]
            if not square.size or square.shape[0] < dy or square.shape[1] < dx:
                logging.warning(f"Empty or undersized square at row {row}, col {col}")
                continue
            # Ensure square is 3-channel BGR
            if len(square.shape) == 2:
                square = cv2.cvtColor(square, cv2.COLOR_GRAY2BGR)
            resized = cv2.resize(square, (224, 224), interpolation=cv2.INTER_AREA)
            name = f'square_r{row}_c{col}.png'
            debug_path = os.path.join(debug_dir, name)
            #cv2.imwrite(debug_path, resized)
            squares.append(resized)
            square_names.append(name)
            square_positions.append((row, col))
    return squares, square_names, square_positions

def generate_fen(squares, square_names, model_session, class_names, frame_idx, debug=True):
    """Generate FEN string from the board squares."""
    board = [['' for _ in range(8)] for _ in range(8)]
    fen_map = {
        'B': 'B', 'K': 'K', 'N': 'N', 'P': 'P', 'Q': 'Q', 'R': 'R',
        'b': 'b', 'k': 'k', 'n': 'n', 'p': 'p', 'q': 'q', 'r': 'r',
        'empty': ''
    }

    for i, (square, name) in enumerate(zip(squares, square_names)):
        row, col = i // 8, i % 8
        piece = match_piece(square, name, model_session, class_names, frame_idx, threshold=0.6, debug=debug)
        if piece and piece in fen_map:
            board[row][col] = fen_map[piece]
            if debug:
                logging.debug(f"Square {name} (r{row}, c{col}): Detected {piece} -> {fen_map[piece]}")
        else:
            board[row][col] = ''
            if debug:
                logging.debug(f"Square {name} (r{row}, c{col}): No piece detected")

    piece_counts = {}
    for row in board:
        for piece in row:
            if piece:
                piece_counts[piece] = piece_counts.get(piece, 0) + 1

    max_counts = {'P': 8, 'p': 8, 'N': 2, 'n': 2, 'B': 2, 'b': 2,
                  'R': 2, 'r': 2, 'Q': 9, 'q': 9, 'K': 1, 'k': 1}
    for piece, count in piece_counts.items():
        if count > max_counts.get(piece, 0):
            logging.warning(f"Invalid piece count for {piece}: {count} exceeds maximum {max_counts.get(piece, 0)}")
            if debug:
                frame_debug_dir = f'./debug_frames/frame_{frame_idx:03d}'
                os.makedirs(frame_debug_dir, exist_ok=True)
                with open(f'{frame_debug_dir}/piece_counts.txt', 'w') as f:
                    f.write(f"Invalid piece counts: {piece_counts}\n")
            return None, board

    fen_rows = []
    for row in board:
        empty = 0
        fen_row = ''
        for square in row:
            if square == '':
                empty += 1
            else:
                if empty > 0:
                    fen_row += str(empty)
                    empty = 0
                fen_row += square
        if empty > 0:
            fen_row += str(empty)
        fen_rows.append(fen_row)

    fen = '/'.join(fen_rows) + ' w KQkq - 0 1'
    if debug:
        logging.info(f"Generated FEN for frame {frame_idx}: {fen}")
        frame_debug_dir = f'./debug_frames/frame_{frame_idx:03d}'
        os.makedirs(frame_debug_dir, exist_ok=True)
        with open(f'{frame_debug_dir}/board_state.txt', 'w') as f:
            f.write(f"FEN: {fen}\n")
            for row in board:
                f.write(f"{row}\n")

    return fen, board

def detect_movement(prev_board, curr_board):
    """Detect differences between two chessboard states and return changed squares."""
    changes = []

    for row in range(8):
        for col in range(8):
            prev_piece = prev_board[row][col]
            curr_piece = curr_board[row][col]
            if prev_piece != curr_piece:
                to_square = f"{chr(97 + col)}{8 - row}"
                
                if prev_piece == '' and curr_piece != '':
                    from_square = None
                    for r in range(8):
                        for c in range(8):
                            if prev_board[r][c] == curr_piece and curr_board[r][c] == '':
                                from_square = f"{chr(97 + c)}{8 - r}"
                                break
                        if from_square:
                            break
                    if from_square:
                        change = f"{from_square}-{to_square}"
                    else:
                        change = f"-{to_square}"
                    changes.append(change)
                
                elif prev_piece != '' and curr_piece == '':
                    from_square = f"{chr(97 + col)}{8 - row}"
                    to_square = None
                    for r in range(8):
                        for c in range(8):
                            if curr_board[r][c] == prev_piece and prev_board[r][c] == '':
                                to_square = f"{chr(97 + c)}{8 - r}"
                                break
                        if to_square:
                            break
                    if to_square:
                        change = f"{from_square}-{to_square}"
                    else:
                        change = f"{from_square}-"
                    changes.append(change)
                
                elif prev_piece != '' and curr_piece != '':
                    from_square = None
                    for r in range(8):
                        for c in range(8):
                            if prev_board[r][c] == curr_piece and curr_board[r][c] == '':
                                from_square = f"{chr(97 + c)}{8 - r}"
                                break
                        if from_square:
                            break
                    if from_square:
                        change = f"{from_square}x{to_square}"
                    else:
                        change = f"x{to_square}"
                    changes.append(change)

    if changes:
        logging.info(f"Detected board changes: {changes}")

    return changes

def annotate_frame(frame, moves, frame_time, points, M):
    """Annotate the frame with detected moves, frame time, and highlight changed squares in green."""
    annotated = frame.copy()
    
    dst = np.array([
        [0, 0],
        [551, 0],
        [551, 551],
        [0, 551]
    ], dtype="float32")
    M_inv = cv2.getPerspectiveTransform(dst, order_points(points))
    
    points_int = points.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(annotated, [points_int], isClosed=True, color=(0, 255, 0), thickness=2)

    square_size = 552 / 8
    highlighted_squares = set()

    for move in moves:
        move_clean = move[1:] if move[0] in 'prnbqkPRNBQK' else move
        try:
            if '-' in move_clean:
                from_square, to_square = move_clean.split('-')
            elif 'x' in move_clean:
                from_square, to_square = move_clean.split('x')
            else:
                logging.warning(f"Invalid move format: {move}")
                continue

            def square_to_coords(square):
                col = ord(square[0]) - ord('a')
                row = 8 - int(square[1])
                if not (0 <= col <= 7 and 1 <= int(square[1]) <= 8):
                    raise ValueError(f"Invalid square: {square}")
                return row, col

            squares_to_highlight = []
            if from_square and from_square != '-':
                squares_to_highlight.append(from_square)
            if to_square and to_square != '':
                squares_to_highlight.append(to_square)

            for square in squares_to_highlight:
                row, col = square_to_coords(square)
                if (row, col) in highlighted_squares:
                    continue
                highlighted_squares.add((row, col))
                
                top_left = np.array([col * square_size, row * square_size], dtype=np.float32)
                bottom_right = np.array([(col + 1) * square_size, (row + 1) * square_size], dtype=np.float32)
                square_pts = np.array([
                    top_left,
                    [bottom_right[0], top_left[1]],
                    bottom_right,
                    [top_left[0], bottom_right[1]]
                ], dtype=np.float32).reshape(-1, 1, 2)

                square_pts_orig = cv2.perspectiveTransform(square_pts, M_inv)
                square_pts_orig = square_pts_orig.astype(int).reshape(-1, 2)
                cv2.polylines(annotated, [square_pts_orig], isClosed=True, color=(0, 255, 0), thickness=3)

        except (ValueError, IndexError) as e:
            logging.warning(f"Failed to parse or highlight move '{move}': {e}")
            continue

    y_offset = 50
    cv2.putText(annotated, f"Time: {frame_time:.2f}s", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    for i, move in enumerate(moves):
        cv2.putText(annotated, f"Move: {move}", (10, y_offset + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return annotated

def main(start_time=0, end_time=10, frame_interval=1.0):
    """Main function to process video, detect chessboard differences, and highlight in a loop."""
    out = None
    cap = None
    try:
        video_path = select_video_file()
        if not video_path:
            logging.error("No video file selected")
            raise ValueError("No video file selected")
        
        if not os.path.exists(video_path):
            logging.error(f"Video file does not exist: {video_path}")
            raise ValueError(f"Video file does not exist: {video_path}")
        
        duration = get_video_duration(video_path)
        if start_time < 0:
            logging.warning(f"Start time ({start_time}s) cannot be negative. Setting to 0.")
            start_time = 0
        if end_time > duration:
            logging.warning(f"Requested end_time ({end_time}s) exceeds video duration ({duration}s). Setting end_time to {duration}s.")
            end_time = duration
        if start_time >= end_time:
            logging.error(f"Start time ({start_time}s) must be less than end time ({end_time}s)")
            raise ValueError("Start time must be less than end time")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Cannot open video file: {video_path}")
            raise ValueError("Cannot open video file")
        
        # Initialize video writer
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            logging.error("Invalid FPS value in video")
            raise ValueError("Invalid FPS value")
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path = "annotated_chess_moves.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps / frame_interval, (frame_width, frame_height))
        
        # Set video to start_time
        start_frame = int(start_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        actual_start_time = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
        if abs(actual_start_time - start_time) > 0.1:
            logging.warning(f"Actual start time ({actual_start_time:.2f}s) differs from requested ({start_time}s)")
        
        # Chessboard detection
        window_name = "Chessboard Detection (Press 'q' to select corners manually)"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        board_detected = False
        board_points = None
        
        # Read first frame at start_time for chessboard detection
        ret, frame = cap.read()
        if not ret:
            logging.error(f"Failed to read frame at start time {start_time}s")
            raise ValueError("Failed to read initial frame")
        
        crop, points = extract_digital_board(frame, debug=True)
        if points is not None:
            display_frame = frame.copy()
            points_int = points.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(display_frame, [points_int], True, (0, 255, 0), 2)
            cv2.putText(display_frame, "Chessboard detected. Press 'a' to accept, 'c' for hardcoded ROI, 'm' for manual ROI", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(window_name, display_frame)
            
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('a'):
                    board_detected = True
                    board_points = points
                    logging.info("Accepted automatically detected chessboard points")
                    break
                elif key == ord('c'):
                    board_detected = True
                    board_points = np.array([
                        [1280, 240],
                        [1879, 240],
                        [1879, 836],
                        [1280, 836]
                    ], dtype="float32")
                    logging.info("Using hardcoded chessboard points")
                    break
                elif key == ord('m'):
                    board_points = select_board_corners(frame)
                    board_detected = True
                    logging.info("Manually selected chessboard points")
                    break
        else:
            logging.info("Automatic chessboard detection failed")
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                board_points = select_board_corners(frame)
                board_detected = True
        
        if not board_detected:
            logging.info("No chessboard detected automatically, using hardcoded points")
            board_points = np.array([
                [1280, 240],
                [1879, 240],
                [1879, 836],
                [1280, 836]
            ], dtype="float32")
        
        # Load model and class names
        model_session = load_model()
        class_names = load_class_names()
        
        # Reset video to start_time for processing
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Process frames in a loop
        fen_results = []
        prev_board = None
        prev_fen = None
        frame_idx = 0
        frame_step = max(1, int(frame_interval * fps))
        end_frame = int(end_time * fps)
        
        while cap.isOpened() and cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame:
            frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"Failed to read frame {frame_num}")
                break
            
            # Process only every frame_step frame
            if frame_num % frame_step != 0:
                continue
                
            frame_time = frame_num / fps
            logging.info(f"Processing frame at {frame_time:.2f} seconds (frame {frame_num})")
            
            global match_result
            match_result = []
            
            try:
                # Warp board
                crop = frame
                board, M = warp_board(crop, board_points)
                frame_debug_dir = f'./debug_frames/frame_{frame_idx:03d}'
                os.makedirs(frame_debug_dir, exist_ok=True)
                cv2.imwrite(f'{frame_debug_dir}/warped_board.png', board)

                # Piece detection
                squares, square_names, square_positions = split_into_squares(board, debug_dir=frame_debug_dir)
                fen, curr_board = generate_fen(squares, square_names, model_session, class_names, frame_idx=frame_idx, debug=True)
                
                if fen is None:
                    logging.warning(f"Skipping frame {frame_idx} at {frame_time:.2f}s due to invalid FEN")
                    out.write(frame)
                    frame_idx += 1
                    continue
                
                logging.info(f"Generated FEN at {frame_time:.2f}s: {fen}")
                
                # Detect differences
                moves = []
                if prev_fen is not None and fen != prev_fen:
                    moves = detect_movement(prev_board, curr_board)
                    logging.info(f"Detected moves at {frame_time:.2f}s: {moves}")
                
                # Highlight and annotate
                annotated_frame = annotate_frame(frame, moves, frame_time, board_points, M)
                
                cv2.imshow(window_name, annotated_frame)
                if moves:
                    cv2.putText(annotated_frame, "Move detected! Press any key to continue.", 
                                (10, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow(window_name, annotated_frame)
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('q'):
                        logging.info("User terminated video display with 'q' key")
                        break
                else:
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('q'):
                        logging.info("User terminated video display with 'q' key")
                        break
                
                out.write(annotated_frame)
                
                fen_results.append((frame_time, fen, moves))
                prev_board = deepcopy(curr_board)
                prev_fen = fen
                frame_idx += 1
            
            except ValueError as e:
                logging.error(f"Processing error for frame at {frame_time:.2f}s: {e}")
                out.write(frame)
                frame_idx += 1
                continue
        
        print("\nFEN and Move Results:")
        for frame_time, fen, moves in fen_results:
            print(f"Time {frame_time:.2f}s: {fen}")
            if moves:
                print(f"  Moves: {', '.join(moves)}")

    except Exception as e:
        logging.error(f"Video processing error: {e}")
        print(f"Failed to process video: {e}")
    finally:
        if out is not None:
            out.release()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main(start_time=200, end_time=220, frame_interval=1)
