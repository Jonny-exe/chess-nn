class Game:
    def __init__(self, game):
        self.game = game

    def get_best_move(self):
        if self.game is None:
            raise Exception("Game is none")
        board = game.board()
        result = (0, None)
        for move in board.legal_moves():
            board_cpy = board
            board_cpy.push(move)
            output = net(board
            if output > result[0]:
                result[0] = output
                result[1] = move
    return result
                
            
