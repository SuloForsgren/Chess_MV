import cv2 as cv
import numpy as np
import chess
from collections import deque

class chess_logic:
    def __init__(self, buffer_len=15, presence_threshold=0.7):
        self.board = chess.Board()
        self.buffer_len = buffer_len
        self.presence_threshold = presence_threshold
        self.square_buffers = {chess.square_name(i): deque(maxlen=self.buffer_len) for i in range(64)}
        self.prev_state = {sq: False for sq in self.square_buffers}

    def check_board(self, squares, debug=False):
        piece_presence = []

        for row in range(8):
            for col in range(8):
                square_img = squares[row][col]

                # Crop center region to avoid edges
                h, w = square_img.shape[:2]
                margin = int(min(h, w) * 0.25)
                cropped = square_img[margin:h - margin, margin:w - margin]
                
                if cropped is None or cropped.size == 0 :
                    print(f"Empty image detected at {row},{col} - skipping")
                    piece_presence.append(False)
                    continue
                gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
                std_dev = np.std(gray)
                edges = cv.Canny(gray, 50, 150)
                edge_pixels = cv.countNonZero(edges)

                has_piece = std_dev > 15 or edge_pixels > 24
                piece_presence.append(has_piece)

        # Update buffers
        for i in range(64):
            square_name = chess.square_name(i)
            self.square_buffers[square_name].append(piece_presence[i])

        # Compute confidence
        square_confidences = {
            sq: sum(buf) / len(buf) if buf else 0
            for sq, buf in self.square_buffers.items()
        }

        # Current binary state
        curr_state = {
            sq: conf > self.presence_threshold
            for sq, conf in square_confidences.items()
        }

        # Detect move
        move = self._detect_move(self.prev_state, curr_state)
        self.prev_state = curr_state.copy()

        if debug:
            self._print_board_presence(curr_state, square_confidences)

        return move  # Either None or a move string like "e2e4"

    def _detect_move(self, prev, curr):
        disappeared = [sq for sq in prev if prev[sq] and not curr[sq]]
        appeared = [sq for sq in curr if not prev[sq] and curr[sq]]

        if len(disappeared) == 1 and len(appeared) == 1:
            return f"{disappeared[0]}{appeared[0]}"
        return None

    def _print_board_presence(self, binary_state, confidences=None):
        for row in range(8):
            rank = row + 1
            for col in range(8):
                file = chr(ord('a') + col)
                sq_index = chess.square(ord(file) - ord('a'), rank - 1)
                square = chess.square_name(sq_index)
                symbol = '●' if binary_state[square] else '·'
                conf = f"({confidences[square]:.2f})" if confidences else ""
                print(f"{square}: {symbol}{conf}".ljust(10), end='')
            print()
        print()