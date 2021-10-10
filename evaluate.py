import value_tables
import numpy as np
def evaluate(board, move):
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
