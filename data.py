#!/usr/bin/python3
import chess
import chess.pgn
import psutil
from collections import defaultdict
import numpy as np
import torch
import value_tables


DEVICE = "cuda:0"
RAM = 8
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
                if psutil.virtual_memory()[2] > 80:
                    if idx == 0:
                        print("Your memory usage is too high")
                        break
                    np.savez(f"{OUTPUT_FILE}.{idx_f}.npz", X, Y, allow_pickle=True)
                    idx_f += 1
                    X = []
                    Y = []

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
                matrix_board = self.board_to_matrix(board)
                X.append(matrix_board)
                Y.append(self.evaluate(matrix_board, move_idx))
                move_idx += 1
            idx += 1

        f = open(f"{OUTPUT_FILE}.batches", "w").write(str(idx_f + 1)).close()
        np.savez(f"{OUTPUT_FILE}.{idx_f}.npz", X, Y, allow_pickle=True)

    def evaluate(self, board, move):
        # 1 = endgame, 0 = middlegame
        phase = 1 if move > 30 else 0

        board = np.array(board).reshape(64, 13)
        piece_eval = 0
        pos = -1
        for piece in board:
            pos += 1
            piece = np.argmax(piece)
            if piece == 6:
                continue
            color = -1 if piece < 6 else +1
            piece = abs(piece - 6)

            piece_table = np.array(value_tables.piece_table[piece][phase])
            if color == -1:
                piece_table = list(np.flipud(piece_table.reshape((8, 8))).reshape(64))

            absolute_eval = color * value_tables.value[phase][piece]
            position_eval = color * piece_table[pos]
            piece_eval += position_eval + absolute_eval
        return piece_eval

    def board_to_matrix(self, board):
        eye = np.eye(13)
        indices = "♚♛♜♝♞♟⭘♙♘♗♖♕♔"
        unicode = board.unicode()
        return [
            [eye[indices.index(c)] for c in row.split()] for row in unicode.split("\n")
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
