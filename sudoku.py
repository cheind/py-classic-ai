__author__ = 'Christoph Heindl'
__copyright__ = 'Copyright 2017'
__license__ = 'BSD'

import numpy as np
import random
from collections import namedtuple

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

Cell = namedtuple('Cell', 'choices row col', verbose=False)

def possibilities(board, row, col):
    i = row // 3
    j = col // 3
    u = np.unique(np.concatenate((board[:, col], board[row, :], board[i*3:i*3+3, j*3:j*3+3].ravel())))
    if u[0] == 0:
        u = u[1:]
    return np.delete([1, 2, 3, 4, 5, 6, 7, 8, 9], u-1)

def solve(board, order):
    rows, cols = np.where(board == 0)
    cells = [Cell(choices=possibilities(board, r, c), row=r, col=c) for r,c in zip(rows, cols)]
    
    if len(cells) == 0:
        return board, order
    
    m = min(cells, key=lambda x:x.choices.shape[0])
    for c in m.choices: # rare need to loop over multiple choices, expect when few fields are known from the beginning.       
        nextboard = board.copy()
        nextorder = order.copy()
        
        nextboard[m.row, m.col] = c
        nextorder[m.row, m.col] = np.max(order) + 1
        
        nextboard, nextorder = solve(nextboard, nextorder)
        if nextboard is not None:  
            return nextboard, nextorder      
    return None, order

board, order = solve(board, np.zeros((9,9), dtype=int))

print(board)
print(order)
