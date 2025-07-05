import cv2 as cv
import numpy as np
import chess

class chess_logic:
    def __init__(self):
        self.board = chess.Board()

    def check_board(self, squares):
        piece_presence = []

        for row in range(8):
            for col in range(8):
                square_img = squares[row][col]

                # Convert to grayscale
                gray = cv.cvtColor(square_img, cv.COLOR_BGR2GRAY)

                # ⬇️ Focus only on center of the square to avoid borders
                h, w = gray.shape
                margin = int(min(h, w) * 0.25)
                cropped = gray[margin:h - margin, margin:w - margin]

                # Heuristic 1: texture variation in center
                std_dev = np.std(cropped)

                # Heuristic 2: edge density in center
                edges = cv.Canny(cropped, 50, 150)
                edge_pixels = cv.countNonZero(edges)

                # Combined heuristic (tuneable thresholds)
                has_piece = std_dev > 10 or edge_pixels > 20

                piece_presence.append(has_piece)

                # Optional: debug per square
                square_name = f"{chr(97 + col)}{8 - row}"
                print(f"{square_name}: std={std_dev:.1f}, edges={edge_pixels}, piece={has_piece}")

        self.print_board_presence(piece_presence)

    def print_board_presence(self, piece_array):
        for i in range(64):
            sq = chess.square_name(i)
            has_piece = piece_array[i]
            print(f"{sq}: {'●' if has_piece else '·'}", end='  ')
            if (i + 1) % 8 == 0:
                print()
        print()
