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
board_img = "img/second.jpg"
img = cv.imread(board_img)
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray_img, (5,5), 0)
edges = cv.Canny(blur, 50, 150)

contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

for point in contours:
    approx = cv.approxPolyDP(point, 0.02 * cv.arcLength(point, True), True)

    if len(approx) == 4 :
        area = cv.contourArea(point)
        if area < 1000 : 
            continue

        corners = approx.reshape(4,2)
        ordered_corners = order_corners(corners)

        (tl, tr, br, bl) = ordered_corners
        width_top = np.linalg.norm(tl - tr)
        width_bot = np.linalg.norm(bl-br)
        height_left = np.linalg.norm(tl-bl)
        height_right = np.linalg.norm(tr-br)

        max_width = int(max(width_top, width_bot))
        max_height = int(max(height_left, height_right))

        aspect_ratio = float(max_width) / max_height


        if 0.9 < aspect_ratio < 1.1 :
            largest_square = ordered_corners
            break
            

if largest_square is not None:
    dst_corners = np.array([
        [0, 0],  # Top-left corner
        [max_width - 1, 0],  # Top-right corner
        [max_width - 1, max_height - 1],  # Bottom-right corner
        [0, max_height - 1]  # Bottom-left corner
    ], dtype="float32")

    img_with_contours = edges.copy()

    matrix = cv.getPerspectiveTransform(largest_square, dst_corners)
    warped = cv.warpPerspective(img, matrix, (max_width, max_height))

    cv.imshow("Detected Chessboard", img_with_contours)
    cv.waitKey(0)
    cv.imshow("Detected Chessboard", warped)
    cv.waitKey(0)

    square_size = warped.shape[0] // 8

    for row in range(8) :
        for col in range(8) :
            x_start = col * square_size
            y_start = row * square_size
            x_end = (col + 1) * square_size
            y_end = (row + 1) * square_size

            grid_cell = warped[y_start:y_end, x_start:x_end]
            cv.rectangle(warped, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)


    resized_image = cv.resize(warped, (600,600))
    cv.imshow("Detected Chessboard", resized_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

else :
    print("No valid squares found")