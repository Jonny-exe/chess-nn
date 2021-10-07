#!/usr/bin/python3
import chess
import chess.pgn
from collections import defaultdict
import numpy as np
import torch


DEVICE = "cuda:0"
class Data:
    result_map = defaultdict(lambda: 0)
    result_map["1-0"] = 1
    result_map["0-1"] = -1
    result_map["1/2-1/2"] = 0

    PIECE_VALUE = dict([
        (0, 0),
        (1, 100), 
        (3, 350), 
        (2, 350), 
        (4, 525), 
        (5, 1000),
        (6, 100),
        ])
    PIECES = dict([
        (0, ""),
        (1, "p"), 
        (3, "b"), 
        (2, "n"), 
        (4, "r"), 
        (5, "q"),
        (6, "k"),
        ])

    def __init__(self):
        pgn = open("data.pgn")
        games = []
        idx = 0
        while 1:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            board = game.board()
            moves = game.mainline_moves()
            result = self.result_map[game.headers["Result"]]
            if idx > 10:
                break
            if idx % 1000 == 0:
                print(idx)
            matrix_board = []
            for move in moves:
                board.push(move)
                matrix_board = self.board_to_matrix(board)
                games.append([matrix_board, result])
            piece_eval =self.evaluate(matrix_board)
            print(board, piece_eval, result)

            idx += 1
        np.save("db.npy", np.array(games), allow_pickle=True)

    def evaluate(self, board):
        board = np.array(board).reshape(64, 13)
        print(board.shape)
        piece_eval = 0
        for piece in board:
            piece = np.argmax(piece)
            color = -1 if piece < 6 else +1
            piece_eval += color * self.PIECE_VALUE[abs(piece - 6)]
        return piece_eval

    def board_to_matrix(self, board):
        eye = np.eye(13)
        indices = '♚♛♜♝♞♟⭘♙♘♗♖♕♔'
        unicode = board.unicode()
        return [
            [eye[indices.index(c)] for c in row.split()]
            for row in unicode.split('\n')
        ]

class DataSet:
    def __init__(self, data):
        self.data = data
        self.X = torch.Tensor([x[0] for x in data]).to(DEVICE).reshape([-1, 1, 8, 8])
        self.Y = torch.Tensor([y[1] for y in data]).to(DEVICE)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        return self.X[index], self.Y[index]

data = Data()
