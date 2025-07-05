import numpy as np
import cv2 as cv

def order_corners(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]     # top-left
    rect[2] = pts[np.argmax(s)]     # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def inset_quad(quad, inset_x=0.06, inset_y=0.06):
    center = np.mean(quad, axis=0)
    adjusted = []
    for pt in quad:
        dx = center[0] - pt[0]
        dy = center[1] - pt[1]
        pt_new = pt + np.array([dx * inset_x, dy * inset_y])
        adjusted.append(pt_new)
    return np.array(adjusted, dtype="float32")

def preprocess_image(board_img):
    img = cv.resize(board_img, (800, int(board_img.shape[0] * 800 / board_img.shape[1])))
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray_img, (5,5), 0)
    thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 5)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv.dilate(thresh, kernel, iterations=2)
    cv.imshow("Detected Chessboard (Edges)", dilated)
    return img, dilated

def draw_grid(img, rows=8, cols=8, color=(0, 255, 0), thickness=2):
    h, w = img.shape[:2]
    dx = w // cols
    dy = h // rows
    for i in range(1, cols):
        x = i * dx
        cv.line(img, (x, 0), (x, h), color, thickness)
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

    once = False
    while True:
        ret, frame = capture.read()
        if not ret or frame is None or frame.size == 0:
            print("Failed to grab frame.")
            break

        try:
            cv.imshow("Live", frame)
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            img, dilated = preprocess_image(frame)
            contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            largest_quad = None
            max_area = 0

            for cnt in contours:
                peri = cv.arcLength(cnt, True)
                approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
                if len(approx) == 4 and cv.isContourConvex(approx):
                    area = cv.contourArea(approx)
                    if area > max_area:
                        max_area = area
                        largest_quad = approx.reshape(4, 2)

            if largest_quad is None:
                print("No board found.")
                continue

            ordered_corners = order_corners(largest_quad)
            play_area_corners = inset_quad(ordered_corners, 0.15, 0.13)

            debug_vis = img.copy()
            cv.polylines(debug_vis, [ordered_corners.astype(int)], True, (0, 255, 0), 2)  # outer
            cv.polylines(debug_vis, [play_area_corners.astype(int)], True, (0, 0, 255), 2)  # shrunk
            cv.imshow("Board Outline (Green=Full, Red=Play Area)", debug_vis)

            # Warp to top-down square
            wA = np.linalg.norm(play_area_corners[2] - play_area_corners[3])
            wB = np.linalg.norm(play_area_corners[1] - play_area_corners[0])
            maxW = max(int(wA), int(wB))

            hA = np.linalg.norm(play_area_corners[1] - play_area_corners[2])
            hB = np.linalg.norm(play_area_corners[0] - play_area_corners[3])
            maxH = max(int(hA), int(hB))

            max_dim = max(maxW, maxH)
            dst = np.array([
                [0, 0],
                [max_dim - 1, 0],
                [max_dim - 1, max_dim - 1],
                [0, max_dim - 1]
            ], dtype="float32")

            M = cv.getPerspectiveTransform(play_area_corners, dst)
            warped = cv.warpPerspective(img, M, (max_dim, max_dim))

            warped_with_grid = draw_grid(warped.copy(), rows=8, cols=8)
            cv.imshow("Warped Chessboard", warped_with_grid)

            # Get squares and annotate
            h, w = warped.shape[:2]
            x_edges = np.linspace(0, w, 9, dtype=int)
            y_edges = np.linspace(0, h, 9, dtype=int)

            squares = []
            for row in range(8):
                row_squares = []
                for col in range(8):
                    x1, x2 = x_edges[col], x_edges[col + 1]
                    y1, y2 = y_edges[row], y_edges[row + 1]
                    square = warped[y1:y2, x1:x2]
                    row_squares.append(square)
                squares.append(row_squares)

            if not once:
                once = True
                for row in squares:
                    for sq in row:
                        print(sq.shape)

            annotated = warped.copy()
            for row in range(8):
                for col in range(8):
                    x1, x2 = x_edges[col], x_edges[col + 1]
                    y1, y2 = y_edges[row], y_edges[row + 1]
                    square_name = f"{chr(97 + col)}{8 - row}"
                    cv.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv.putText(annotated, square_name, (x1 + 5, y1 + 15),
                               cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            cv.imshow("Annotated Chessboard", annotated)

            if cv.waitKey(1) & 0xFF == ord('s'):
                break

        except Exception as e:
            print("An exception occurred:", e)
            continue

    capture.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
