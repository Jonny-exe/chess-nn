#!/usr/bin/python3
import chess
import chess.pgn
from collections import defaultdict
import numpy as np


class Data:
    result_map = defaultdict(lambda: 0)
    result_map["1-0"] = 1
    result_map["0-1"] = -1
    result_map["1/2-1/2"] = 0

    def __init__(self):
        pgn = open("lichess.pgn")
        games = []
        idx = 0
        while 1:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            board = game.board()
            moves = game.mainline_moves()
            result = self.result_map[game.headers["Result"]]
            if idx % 1000 == 0:
                print(idx)
            for move in moves:
                board.push(move)
                games.append([self.board_to_matrix(board), result])
            idx += 1
        np.save("data.npy", np.array(games), allow_pickle=True)
            

    def board_to_matrix(self, board):
        indices = '♚♛♜♝♞♟⭘♙♘♗♖♕♔'
        unicode = board.unicode()
        return [
            [indices.index(c)-6 for c in row.split()]
            for row in unicode.split('\n')
        ]

      
data = Data()
