import numpy as np
from sys import float_info

class Board:
    '''Represents the current state of the game.'''

    WIN_SCORE = 1e4

    def __init__(self, state=None, player=0, score=None):
        '''Initialize board.'''

        if state is None:
            self.state = np.zeros((6,7), dtype='b')
        else:
            self.state = state.copy()
        if score is None:
            self.score = [0., 0.]
        else:
            self.score = np.copy(score)
        self.player = player

    def copy(self):
        '''Returns a deep-copy of the board.'''
        return Board(state=self.state, player=self.player, score=self.score)

    def __repr__(self):
        '''Debug representation of board state.'''
        return f'Board(score={self.score}, finished={self.finished}, next-player={self.player})\n{self.state}'

    def __str__(self):
        '''Stringify board.'''
        mapped = ['x', ' ', '#']
        s = '\n'.join(''.join(mapped[cell+1] for cell in row) for row in self.state)
        s = s + '\n' + '-'*self.state.shape[1]
        s = s + '\n' + ''.join([str(c) for c in np.arange(0, self.state.shape[1])])
        return s

    def move(self, col):
        '''Places a disc in given column for current player.'''
        assert col in self.possible_moves, f'Cannot place in column {col}.'

        row = self.state.shape[0] - np.count_nonzero(self.state[:, col]) - 1
        self.last_move = [row, col]
        self.state[row, col] = -1 if self.player == 0 else 1
        self.score = self._compute_score(row, col)
        self.player = (self.player + 1) % 2

    def copy_move(self, col):
        b = self.copy()
        b.move(col)
        return b

    @property
    def possible_moves(self):
        '''Returns the available columns to place a disc.'''
        return np.where(self.state[0,:] == 0)[0]

    @property
    def finished(self):
        '''Returns true if the game has finished, false otherwise.'''
        return abs(self.score[0]) == Board.WIN_SCORE or self.possible_moves.shape[0] == 0

    def _compute_score(self, row, col):
        '''Returns the score for a given move.
        
        This is the evaluation function for the agent and is
        responsible for how meaningful its moves are. The function
        assigns Board.WIN_SCORE in case any player wins or
        computes a heuristic for the move by counting the
        number of slots per player for every combination of four.        
        '''
        def score_strip(x):
            scores = np.zeros(2)
            for f in [x[i:i+4] for i in range(x.shape[0] - 3)]:
                c = np.bincount(f, minlength=3)
                if c[0] == 4:
                    return [Board.WIN_SCORE, -Board.WIN_SCORE]
                elif c[2] == 4:
                    return [-Board.WIN_SCORE, Board.WIN_SCORE]
                scores[0] += c[0] - c[2]
                scores[1] += c[2] - c[0]
            return scores
        
        # For each direction horizontal, vertical, 2xdiagonal

        strips = [
            self.state[row],
            self.state[:, col],
            np.diagonal(self.state, offset=col-row),
            np.diagonal(self.state[:, ::-1], offset=self.state.shape[0]-col-row)
        ]

        total_score = np.zeros(2)
        for strip in strips:
            s = score_strip(strip + 1)
            if abs(s[0]) == Board.WIN_SCORE:
                return s
            total_score += s
        return total_score
        
class Agent:
    '''Artificial agent playing connect-four.'''

    def __init__(self, board, player, max_depth=4):
        self.board = board
        self.max_depth = max_depth
        self.player = player

    def move(self):
        best = self.maximize(self.board, -float_info.max, float_info.max, self.max_depth)
        self.board.move(best[1])

    def maximize(self, board, alpha, beta, depth):
        if self.terminal(board, depth):
            return self.utility(board)
        else:
            v = [-float_info.max, None]
            for m in board.possible_moves:
                r = self.minimize(board.copy_move(m), alpha, beta, depth - 1)
                if r[0] > v[0]:
                    v[0] = r[0]
                    v[1] = m
                    alpha = v[0]
                if alpha >= beta:
                    return v
            return v
    
    def minimize(self, board, alpha, beta, depth):
        if self.terminal(board, depth):
            return self.utility(board)
        else:
            v = [float_info.max, None]
            for m in board.possible_moves:
                r = self.maximize(board.copy_move(m), alpha, beta, depth - 1)
                if r[0] < v[0]:
                    v[0] = r[0]
                    v[1] = m
                    beta = v[0]
                if alpha >= beta:
                    return v
            return v

    def terminal(self, board, depth):
        return depth == 0 or board.finished

    def utility(self, board):
        return [board.score[self.player], board.last_move[1]]

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='4-connect')
    parser.add_argument('--player', type=int, default=0, help='Player to play with - 0 starts.')
    parser.add_argument('--depth', type=int, default=4, help='AI lookahead depth')
    args = parser.parse_args()

    b = Board()
    o = Agent(b, player=(args.player + 1 % 2), max_depth=args.depth)
    while not b.finished:
        print(b)
        if b.player == args.player:
            b.move(int(input('which column? ')))
        else:
            print('Computing...')
            o.move()

    print(b)
    print('Game finished')