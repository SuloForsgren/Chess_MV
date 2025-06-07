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
    img = cv.imread(board_img)
    img = cv.resize(img, (800, int(img.shape[0] * 800 / img.shape[1])))
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
    board_img = "img/start2.jpg" # Define the image
    img, edges = preprocess_image(board_img)

    # Finds contours of the chessboard likely the play area or whole chessboard borders
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largest_square = max(contours, key=cv.contourArea) # Detect the largest square found

    # Get the play area and 4 corners of that
    rect = cv.minAreaRect(largest_square)
    box = cv.boxPoints(rect).astype("float32")
    
    # Order the corners consistently in this order: top-left, top-right, bottom-right, bottom-left
    ordered_box = order_corners(box)

    # Compute width of the play area by choosing the largest distance between bottom line and top line 
    widthA = np.linalg.norm(ordered_box[2] - ordered_box[3])
    widthB = np.linalg.norm(ordered_box[1] - ordered_box[0])
    maxWidth = max(int(widthA), int(widthB))

    # Same for the height
    heightA = np.linalg.norm(ordered_box[1] - ordered_box[2])
    heightB = np.linalg.norm(ordered_box[0] - ordered_box[3])
    maxHeight = max(int(heightA), int(heightB))

    # Define destination points for the new (warped) image
    dst_corners = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Compute and apply the new perspective transform matrix from the ordered corners to the destination rectangle  
    M = cv.getPerspectiveTransform(ordered_box, dst_corners)
    warped = cv.warpPerspective(img, M, (maxWidth, maxHeight))

    # Draw grid over the warped image
    warped_with_grid = draw_grid(warped.copy(), rows=8, cols=8)

    # Show everything
    img_contours_gray = img.copy()
    cv.polylines(img_contours_gray, [box.astype(int)], isClosed=True, color=(0, 255, 0), thickness=3)

    # Show the image with drawn contours and wait user input to close all windows
    cv.imshow("Detected Chessboard Corners", img_contours_gray)
    cv.imshow("Warped Chessboard", warped_with_grid)
    #cv.waitKey(0)  # Comment out to skip showing images 
    cv.destroyAllWindows()

    # Cell detection
    cells = extract_cells(warped)
    piece_array = []

    # Check if playing from left to right or from bot to top
    # Can be used later again for move checks
    for index, cell in enumerate(cells):
        if detect_piece(cell):
            if index == 3 or index == 4:
                piece_array.append(True)
        else:
            if index == 3 or index == 4:
                piece_array.append(False)

    # Detect if board is set from left to right or bot to top then rotate the image
    # left-right = True, bot-top = False 
    if detect_start_pos(piece_array) :
        print("R\nO\nT\nA\nT\nE\n")
        rotated = cv.rotate(warped, cv.ROTATE_90_CLOCKWISE)
        cells = extract_cells(rotated)
        

    piece_array.clear()
    for index, cell in enumerate(cells):
        if detect_piece(cell):
            piece_array.append(True)
            print("True")
        else:
            piece_array.append(False)
            print("False")


    # Use chess_logic class to create an object for the board 
    chess = chess_detection.chess_logic()
    if chess.check_board(piece_array):     # Check current board status
        print("Everything is fine!")
    else :
        print("Pieces at wrong places.")


if __name__ == "__main__":
    main()
