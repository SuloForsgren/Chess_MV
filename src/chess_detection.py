import chess

class chess_logic:
    def __init__(self):
        self.board = chess.Board()
        
    def check_board(self, piece_array):
        for i in range(64):
            piece = self.board.piece_at(i)
            if piece != None :
                if piece_array[i] :
                    continue
                else:
                    return False
            else:
                if piece_array[i] == False :
                    continue
                else:
                    return False
                
        return True