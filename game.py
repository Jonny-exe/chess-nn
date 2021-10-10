import chess
import chess.svg
import base64
from evaluate import evaluate
from data import Data

class Game:
    def __init__(self, game, use_net: bool):
        self.move_idx = 0
        if game is None:
            self.board = chess.Board()
        else:
            self.board = game.board()
        if use_net:
            self.net = net
        self.use_net = use_net

    def play_best_move(self):
        best_move = self.get_best_move()
        print(self.board)
        self.board.push(best_move)
        self.move_idx += 1

    def get_best_move(self):
        result = [0, None]
        turn = +1 if self.board.turn == chess.WHITE else -1
        for move in self.board.legal_moves:
            print(move)
            board_cpy = self.board.copy()
            board_cpy.push(move)
            print(board_cpy)
            if self.use_net:
                output = net(self.board)
            else:
                matrix_board = Data.board_to_matrix(board_cpy)
                output = turn * evaluate(matrix_board, self.move_idx)
                print(turn * output)
            if output > result[0] or result[1] is None:
                result[0] = output
                result[1] = move
        return result[1]

    def get_svg(self):
        svg = base64.b64encode(chess.svg.board(board=self.board, size=400).encode("utf-8")).decode("utf-8")
        return svg
                    
            
