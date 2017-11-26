import numpy as np
import copy
from enum import Enum
from random import shuffle
import sys

class Result(Enum):
    PLAYER0_WINS = 0
    PLAYER1_WINS = 1
    UNDECIDED = 2
    DRAW = 3

class StripScore:

    def eval(self, board):
        strips = board.strips(board.last_move)
        total_scores = np.zeros(2)
        for strip in strips:
            r, s = self.score(strip)
            if r != Result.UNDECIDED:
                return r, s
            total_scores += s
        if board.available_moves.shape[0] == 0:
            return Result.DRAW, total_scores
        else:
            return Result.UNDECIDED, total_scores

    def score(self, x):
        scores = np.zeros(2)
        for f in (x[i:i+4] for i in range(x.shape[0] - 3)):
            c = np.bincount(f+1, minlength=3).astype(float)
            if c[0] == 4:
                return Result.PLAYER0_WINS, [10000., -10000.]
            elif c[-1] == 4:
                return Result.PLAYER1_WINS, [-10000., 10000.]
            elif c[0] > 0 and c[-1] == 0:
                scores += [c[0]**2, -c[0]**2]
            elif c[0] == 0 and c[-1] > 0:
                scores += [-c[-1]**2, c[-1]**2]
        return Result.UNDECIDED, [scores[0], scores[-1]]


class Board:

    def __init__(self, player=0, state=None, scorefnc=StripScore()):
        if state is None:
            self.state = np.zeros((6,7), dtype=int)
        else:
            self.state = np.copy(state)
        self.player = player
        self.available_moves = self._list_moves()
        self.last_move = None
        self.result = Result.UNDECIDED
        self.score = [0, 0]
        self.scorefnc = scorefnc
    
    def copy(self):
        '''Deep copy board.'''
        return copy.deepcopy(self)

    def __repr__(self):
        '''Debug representation of board state.'''
        return f'Board(shape={self.state.shape}, color={self.color})\n{self.state}'

    def __str__(self):
        mapped = ['x', ' ', '#']
        s = '\n'.join(''.join(mapped[cell+1] for cell in row) for row in self.state)
        s = s + '\n' + '-'*self.state.shape[1] + '\n' + ''.join([str(c) for c in np.arange(0, self.state.shape[1])])
        return s

    def move(self, col):
        '''Place a disc in given column.'''
        assert col in self.available_moves, f'Cannot place in column {col}.'

        row = self.state.shape[0] - np.sum(np.abs(self.state[:, col])) - 1
        self.state[row, col] = -1 if self.player == 0 else 1
        self.last_move = (row, col)
        self.available_moves = self._list_moves()
        self.result, self.score = self.scorefnc.eval(self)
        self.player = (self.player + 1) % 2

    def copy_move(self, col):
        b = self.copy()
        b.move(col)
        return b

    def strips(self, move):
        row, col = move
        return [
            self.state[row],
            self.state[:, col],
            np.diagonal(self.state, offset=col-row),
            np.diagonal(self.state[:, ::-1], offset=self.state.shape[0]-col-row)
        ]

    def _list_moves(self):
        '''Returns the available columns to place next disc.'''
        return np.where(np.sum(np.abs(self.state), axis=0) < self.state.shape[0])[0]

class MinimaxPlayer:

    def __init__(self, board, depth=6):
        self.board = board
        self.max_depth = depth

    def move(self):
        self.player = self.board.player
        rated_moves = [(self.minimize(self.board.copy_move(m), -sys.float_info.max, sys.float_info.max, self.max_depth), m) for m in self.board.available_moves]
        shuffle(rated_moves)
        best = max(rated_moves, key=lambda x:x[0])[1]
        self.board.move(best)

    def maximize(self, board, alpha, beta, depth):
        if self.terminal(board, depth):
            return self.utility(board)
        else:
            v = -sys.float_info.max
            for m in board.available_moves:
                v = max(v, self.minimize(board.copy_move(m), alpha, beta, depth - 1))
                if v >= beta:
                    return v
                alpha = max(v, alpha)
            return v
    
    def minimize(self, board, alpha, beta, depth):
        if self.terminal(board, depth):
            return self.utility(board)
        else:
            v = sys.float_info.max
            for m in board.available_moves:
                v = min(v, self.maximize(board.copy_move(m), alpha, beta, depth - 1))
                if v <= alpha:
                    return v
                beta = min(v, beta)
            return v

    def terminal(self, board, depth):
        return board.result != Result.UNDECIDED or depth == 0

    def utility(self, board):
        return board.score[self.player]

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='4-connect')
    parser.add_argument('--player', type=int, default=0, help='Human player starts (0) or AI starts(1)')
    parser.add_argument('--depth', type=int, default=4, help='AI lookahead depth')
    args = parser.parse_args()

    b = Board()
    o = MinimaxPlayer(b, depth=args.depth)
    while b.result == Result.UNDECIDED:
        print(b)
        if b.player == args.player:
            b.move(int(input('which column? ')))
        else:
            print('Computing...')
            o.move()

    print(b)
    print('Game finished')
    