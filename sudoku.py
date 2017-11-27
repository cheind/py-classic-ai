__author__ = 'Christoph Heindl'
__copyright__ = 'Copyright 2017'
__license__ = 'BSD'

import numpy as np

board = np.array([
    0, 0, 9, 4, 8, 0, 0, 0, 0,
    0, 0, 0, 0, 6, 3, 2, 0, 0,
    0, 0, 6, 0, 7, 0, 4, 3, 8,
    0, 0, 0, 0, 0, 0, 7, 0, 6,
    5, 9, 0, 0, 0, 7, 0, 0, 0, 
    7, 0, 8, 0, 4, 9, 0, 1, 3, 
    0, 0, 0, 0, 0, 0, 6, 8, 7,
    0, 0, 0, 8, 9, 1, 0, 0, 0,
    0, 5, 3, 0, 0, 6, 9, 0, 0,
], dtype='b').reshape((9,9))

def possibilities(board, row, col):
    i = row // 3
    j = col // 3
    u = np.unique(np.concatenate((board[:, col], board[row, :], board[i*3:i*3+3, j*3:j*3+3].ravel())))
    if u[0] == 0:
        u = u[1:]
    return np.delete([1, 2, 3, 4, 5, 6, 7, 8, 9], u-1)

def solve(board):
    rows, cols = np.where(board == 0)
    moves = [(possibilities(board, r, c), r, c) for r,c in zip(rows, cols)]
    moves.sort(key=lambda x:x[0].shape[0])

    if len(moves) == 0:
        return board

    board[moves[0][1], moves[0][2]] = moves[0][0][0]
    return solve(board)

print(solve(board))
