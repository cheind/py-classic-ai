__author__ = 'Christoph Heindl'
__copyright__ = 'Copyright 2017'
__license__ = 'BSD'

import numpy as np
import random
from collections import namedtuple

board = np.array([
    0, 0, 1, 0, 0, 6, 3, 5, 0,
    0, 0, 6, 5, 0, 8, 0, 0, 7,
    2, 0, 0, 0, 0, 0, 0, 0, 8,
    0, 0, 5, 0, 0, 0, 0, 9, 2,
    0, 1, 0, 0, 5, 0, 0, 0, 0, 
    6, 9, 0, 0, 0, 0, 0, 4, 0, 
    0, 0, 0, 0, 0, 1, 0, 7, 6,
    0, 0, 0, 3, 6, 0, 0, 0, 0,
    4, 0, 0, 7, 0, 0, 1, 0, 0,
], dtype='b').reshape((9,9))

def init_state(board):
    state = {
        'board' : np.zeros((9,9), dtype='b'),
        'taken' : np.zeros((9,9,9), dtype='b')
    }
    it = np.nditer(board, flags=['multi_index'])
    while not it.finished:
        if it[0] != 0:
            update_state(state, it.multi_index[0], it.multi_index[1], it[0])
        it.iternext()
    return state

def clone_state(state):
    return {
        'board' : state['board'].copy(),
        'taken' : state['taken'].copy(),
    }


def update_state(state, row, col, v):
    state['board'][row, col] = v
    
    i = (row // 3) * 3
    j = (col // 3) * 3
    state['taken'][row, col] = True             # all digits taken
    state['taken'][row, :, v-1] = True          # v taken along row
    state['taken'][:, col, v-1] = True          # v taken along col
    state['taken'][i:i+3, j:j+3, v-1] = True    # v taken along subgrid

    fillin(state)

    return state

def possibilities(state, row, col):    
    avail = np.where(state['taken'][row, col] == 0)[0] + 1
    if avail.shape[0] < 2:
        return avail
    
    free = state['board'] == 0
    mask = np.zeros((9,9), dtype=bool)

    def reducebyopt(mask, avail):
        tk = state['taken'][mask & free]
        tk = np.where(tk.all(0))[0] + 1
        return np.intersect1d(avail, tk)

    i = (row // 3) * 3
    j = (col // 3) * 3    
    mask[i:i+3, j:j+3] = True
    mask[row, col] = False
    ravail = reducebyopt(mask, avail)
    if (ravail.shape[0] == 1):
        return ravail

    mask.fill(False)
    mask[row, :] = True
    mask[row, col] = False
    ravail = reducebyopt(mask, avail)
    if (ravail.shape[0] == 1):
        return ravail

    mask.fill(False)
    mask[:, col] = True
    mask[row, col] = False
    ravail = reducebyopt(mask, avail)
    if (ravail.shape[0] == 1):
        return ravail

    return avail

def fillin(state):
    for i in range(3):
        for j in range(3):
            sel = np.array(np.where(state['taken'][i*3:i*3+3,j*3:j*3+3].transpose((2,0,1)) == 0))

            for k in range(9):
                n = sel[1:, sel[0] == k]
                if n.shape[1] < 2:
                    continue
                
                if ((n[0,0]-n[0,:]) == 0).all():
                    # along row
                    state['taken'][i*3 + n[0,0], :, k] = True
                    state['taken'][i*3 + n[0,:], j*3 + n[1,:], k] = False
                elif ((n[1,0]-n[1,:]) == 0).all():
                    # along col
                    state['taken'][:, j*3 + n[1,0], k] = True
                    state['taken'][i*3 + n[0,:], j*3 + n[1,:], k] = False
                
    


def solve(state):
    stack = [state]

    while len(stack) > 0:
        state = stack.pop()

        rows, cols = np.where(state['board'] == 0)
        cells = [(possibilities(state, r, c), r, c) for r,c in zip(rows, cols)]

        if len(cells) == 0:
            return state
    
        m = min(cells, key=lambda x:x[0].shape[0])
        print(f'{m}')
        if m[0].shape[0] == 1:
            update_state(state, m[1], m[2], m[0][0])
            stack.append(state)
        else:          

            print(state['board']) 
            print(sorted(cells, key=lambda x: len(x[0])))
            for c in m[0]: # rare need to loop over multiple choices, expect when few fields are known from the beginning.       
                clone = clone_state(state)
                update_state(clone, m[1], m[2], c)
                stack.append(clone)

                
    return None


print(solve(init_state(board)))