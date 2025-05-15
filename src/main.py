import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
 
def order_corners(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]     # top-left
    rect[2] = pts[np.argmax(s)]     # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect

def preprocess_image(board_img) :
    img = cv.imread(board_img)
    img = cv.resize(img, (800, int(img.shape[0] * 800 / img.shape[1])))

    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray_img, (5,5), 0)
    edges = cv.Canny(blur, 50, 150)

    resized_image = cv.resize(edges, (600,600))
    cv.imshow("Detected Chessboard", resized_image)

    return img, edges

def main() :
    largest_square = None

    board_img = "img/origin.jpg"
    img, edges = preprocess_image(board_img)

    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    largest_square = max(contours, key=cv.contourArea)

    rect = cv.minAreaRect(largest_square)
    box = cv.boxPoints(rect)
    box = box.astype("float32")

    ordered_box = order_corners(box)

    widthA = np.linalg.norm(ordered_box[2] - ordered_box[3])
    widthB = np.linalg.norm(ordered_box[1] - ordered_box[0])
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(ordered_box[1] - ordered_box[2])
    heightB = np.linalg.norm(ordered_box[0] - ordered_box[3])
    maxHeight = max(int(heightA), int(heightB))

    dst_corners = np.array([
        [0, 0],  # Top-left corner
        [maxWidth - 1, 0],  # Top-right corner
        [maxWidth - 1, maxHeight - 1],  # Bottom-right corner
        [0, maxHeight - 1]  # Bottom-left corner
    ], dtype="float32")

    img_with_contours = edges.copy()

    matrix = cv.getPerspectiveTransform(ordered_box, dst_corners)
    warped = cv.warpPerspective(img, matrix, (maxWidth, maxHeight))

    img_contours_gray = img.copy()
    cv.polylines(img_contours_gray, [box.astype(int)], isClosed=True, color=(0, 255, 0), thickness=3)

    cv.imshow("Detected Chessboard Corners", img_contours_gray)
    cv.imshow("Warped Chessboard", warped)
    cv.waitKey(0)
    cv.destroyAllWindows()

    print("Chessboard corner points (ordered):")
    print(ordered_box)

if __name__ == "__main__" :
    main()