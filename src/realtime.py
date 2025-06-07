#import chess_detection
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


def main():
    cam_index = 0
    capture = cv.VideoCapture(cam_index)

    if not capture.isOpened():
            print("Error: Cannot access webcam.")
            return

    while True:
        ret, frame = capture.read()
        if not ret or frame == None or frame.size == 0:
            print("Failed to grab frame.")
            break

        
        # Resize to a fixed width
        img = cv.resize(frame, (800, int(frame.shape[0] * 800 / frame.shape[1])))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (5, 5), 0)
        edges = cv.Canny(blur, 50, 150)

        # Debug
        cv.imshow("Show the edges for debugging", edges)

        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        

        suitable_corners = []
        for cnt in contours :
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4:
                quad = approx.reshape(4, 2)
                area = cv.contourArea(quad)
                suitable_corners.append((area, quad))

        _, board_quad = max(suitable_corners, key=lambda x: x[0])
        board_quad = board_quad.astype("float32")

        ordered_corners = order_corners(board_quad)

        wA = np.linalg.norm(ordered_corners[2] - ordered_corners[3])
        wB = np.linalg.norm(ordered_corners[1] - ordered_corners[0])
        maxW = max(int(wA), int(wB))

        hA = np.linalg.norm(ordered_corners[1] - ordered_corners[2])
        hB = np.linalg.norm(ordered_corners[0] - ordered_corners[3])
        maxH = max(int(hA), int(hB))

        dst = np.array([
            [0, 0],
            [maxW - 1, 0],
            [maxW - 1, maxH - 1],
            [0, maxH - 1]
        ], dtype="float32")

        M = cv.getPerspectiveTransform(ordered_corners, dst)
        warped = cv.warpPerspective(img, M, (maxW, maxH))

        warped_with_grid = draw_grid(warped.copy(), rows=8, cols=8)

        debug_vis = img.copy()
        cv.polylines(debug_vis, [ordered_corners.astype(int)], True, (0, 255, 0), 3)
        cv.imshow("Original image with board outline", debug_vis)

        cv.imshow("Warped image", warped_with_grid)
        if cv.waitKey(1) & 0xFF == ord('s'):
            break

    capture.release()
    cv.destroyAllWindows()



if __name__ == "__main__":
    main()
