import chess_detection
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Order the contours to certain order (tl, tr, br, bl)
def order_corners(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]     # top-left
    rect[2] = pts[np.argmax(s)]     # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

# Settings for image detection
# Grayscale to detect lines and blur to filter noise
def preprocess_image(board_img):
    img = cv.resize(board_img, (800, int(board_img.shape[0] * 800 / board_img.shape[1])))
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray_img, (5,5), 0)
    edges = cv.Canny(blur, 50, 150)

    # Optional preview
    resized_image = cv.resize(edges, (600,600))
    cv.imshow("Detected Chessboard (Edges)", resized_image)

    return img, edges

# Draws 8x8(64 squares) grid to the chessboard image
def draw_grid(img, rows=8, cols=8, color=(0, 255, 0), thickness=2):
    h, w = img.shape[:2]
    dx = w // cols
    dy = h // rows

    # Vertical lines
    for i in range(1, cols):
        x = i * dx
        cv.line(img, (x, 0), (x, h), color, thickness)

    # Horizontal lines
    for i in range(1, rows):
        y = i * dy
        cv.line(img, (0, y), (w, y), color, thickness)

    return img

# Extract cells to an array for future use
def extract_cells(warped):
    rows = 8
    cols = 8
    h, w = warped.shape[:2]
    
    cell_height = h // rows
    cell_width = w // cols 
    cells = []

    for row in range(rows):
        for col in range(cols):
            y1 = row * cell_height
            y2 = (row + 1) * cell_height
            x1 = col * cell_width
            x2 = (col + 1) * cell_width
            cell = warped[y1:y2, x1:x2]
            cells.append(cell)

    return cells

# Detect pieces from cells by counting pixel color difference
def detect_piece(cell):
    crop_ratio = 0.5
    diff_threshold = 20
    percent_threshold = 0.15

    gray = cv.cvtColor(cell, cv.COLOR_BGR2GRAY)

    #cv.imshow("gray piece", gray)
    #cv.waitKey(0)

    h, w = gray.shape
    cx, cy = w // 2, h // 2
    half_crop_w, half_crop_h = int(w * crop_ratio / 2), int(h * crop_ratio / 2)
    center_crop = gray[cy - half_crop_h:cy + half_crop_h, cx - half_crop_w:cx + half_crop_w]
    
    # Find dominant pixel intensity (median)
    dominant_value = np.median(center_crop)

    # Calculate difference from dominant value for each pixel
    diff = np.abs(center_crop.astype(int) - int(dominant_value))
    count_diff_pixels = np.sum(diff > diff_threshold)       # Count pixels that differ from the threshold
    percent_diff = count_diff_pixels / center_crop.size     # Calculate percentage of differing pixels

    return percent_diff > percent_threshold

def detect_start_pos(piece_array):
    if piece_array[0] == True and piece_array[1] == True :
        return False
    elif piece_array[0] == False and piece_array[1] == False :
        return True

def main():
    cam_index = 0
    capture = cv.VideoCapture(cam_index)

    if not capture.isOpened():
            print("Error: Cannot access webcam.")
            return

    while True:
        ret, frame = capture.read()
        if not ret:
            print("Failed to grab frame.")
            break

        
        img, edges = preprocess_image(frame)
        cv.imshow('Webcam Feed', edges)
        # Finds contours of the chessboard likely the play area or whole chessboard borders
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            print("No contours")
            break
        else:
            print(f"{contours}")
            # 1) pick the “largest” contour
            largest_idx = np.argmax([cv.contourArea(c) for c in contours])
            board_contour = contours[largest_idx]

            # 2) visualize it on the edges (or on the color image) to confirm
            debug_vis = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
            cv.drawContours(debug_vis, contours, largest_idx, (0,0,255), 2)
            cv.imshow("Which Contour Was Chosen?", debug_vis)



if __name__ == "__main__":
    main()
