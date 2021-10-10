#!/usr/bin/python3
import chess
import chess.pgn
import psutil
from collections import defaultdict
import numpy as np
import torch
import value_tables
from evaluate import evaluate


DEVICE = "cuda:0"
RAM = 80
OUTPUT_FILE = "new"


class Data:
    result_map = defaultdict(lambda: 0)
    result_map["1-0"] = 1
    result_map["0-1"] = -1
    result_map["1/2-1/2"] = 0

    PIECE_VALUE = dict(
        [
            (0, 0),
            (1, 100),
            (3, 350),
            (2, 350),
            (4, 525),
            (5, 1000),
            (6, 100),
        ]
    )
    PIECES = dict(
        [
            (0, ""),
            (1, "p"),
            (3, "b"),
            (2, "n"),
            (4, "r"),
            (5, "q"),
            (6, "k"),
        ]
    )

    def __init__(self):
        pgn = open("data.pgn")
        X = []
        Y = []
        idx = 0
        idx_f = 0
        while 1:
            # Memory usage is too high
            if idx % 1000 == 0:
                print(psutil.virtual_memory()[2])
                if psutil.virtual_memory()[2] > 40:
                    if idx == 0:
                        print("Your memory usage is too high")
                        break
                    np.savez(f"{OUTPUT_FILE}.{idx_f}.npz", X, Y, allow_pickle=True)
                    idx_f += 1
                    X = []
                    Y = []
                    break

            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            board = game.board()
            moves = game.mainline_moves()
            result = self.result_map[game.headers["Result"]]
            if idx % 1000 == 0:
                print(idx)
            matrix_board = []
            move_idx = 0
            for move in moves:
                board.push(move)
                matrix_board = board_to_matrix(board)
                X.append(matrix_board)
                Y.append(evaluate(matrix_board, move_idx))
                move_idx += 1
            idx += 1

        # np.savez(f"{OUTPUT_FILE}.{idx_f}.npz", X, Y, allow_pickle=True)

    def board_to_matrix(board):
        eye = np.eye(13)
        indices = "♚♛♜♝♞♟⭘♙♘♗♖♕♔"
        unicode = board.unicode()
        return [
            [eye[indices.index(c)] for c in row.split()] for row in unicode.split("\n")
        ]

class DataSet:
    def __init__(self, X, Y):
        self.X = torch.Tensor(X).to(DEVICE).reshape([-1, 1, 64, 13])
        self.Y = torch.Tensor(Y).to(DEVICE)

    def __len__(self):
        assert self.X.shape[0] == self.Y.shape[0]
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

if __name__ == "__main__":
    data = Data()
