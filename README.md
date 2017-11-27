# py-classic-ai
Various classic artificial intelligence algorithms applied to common problems.

## connectfour.py

Connect Four is a two player game where one player attempts to connect 4 discs (horizontally, vertically or diagonally) while
preventing the other player from doing so. The code allows you to play the game against a artificial agent utilizing a depth limited adversarial [MiniMax](https://en.wikipedia.org/wiki/Minimax) search to determine its next move. In particular, the implementation is based on [Negamax](https://en.wikipedia.org/wiki/Negamax) and uses [Alpha-Beta pruning](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning) to quickly discard search regions.

## sudoku.py

Sudoku is logical number puzzle game. The objective is to fill a 9x9 grid with digits such that each digit from 1 to 9 occurs exactly once along each row, each column and each 3x3 sub-grid. According to [this article](https://en.wikipedia.org/wiki/Sudoku#Mathematics_of_Sudoku) the fewest number of cells filled for a unique solution is 17. The solver implemented is based on [greedy search](https://en.wikipedia.org/wiki/Greedy_algorithm) utilizing a depth-first search traversal and a heuristic that sorts potential moves based on the number of unconstrained neighbor cells for a given cell. The more numbers along the cell's row/column/sub-grid are known, the better the heuristic. The heuristic works well in practice and leads to almost always a solution when reaching the first leaf. Occasionally, when only few cells are filled from the beginning, multiple digit choices are available for a single cell and the algorithm might fail to choose the correct one. However it will backtrack to the correct solution if there is one.