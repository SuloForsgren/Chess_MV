import chess

class chess_logic:
    def __init__(self):
        self.board = chess.Board()
        
    #def check_board(self, piece_array):
    #    for i in range(64):
    #        piece = self.board.piece_at(i)
    #        if bool(piece) != piece_array[i]:
    #            return False
    #    return True

    #def print_board_presence(self, piece_array):
    #    # chess.square_name(i) gives you 'a1', 'b1', … 'h8'
    #    for i in range(64):
    #       sq = chess.square_name(i)
    #        has_piece = piece_array[i]
    #        print(f"{sq}: {'●' if has_piece else '·'}")
    #    print()  # blank line at end


    def check_board(self) :
        print (self.board)