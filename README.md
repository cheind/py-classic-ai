# py-classic-ai
Various classic artificial intelligence algorithms applied to common problems.

## connectfour.py

Connect Four is a two player game where one player attempts to connect 4 discs (horizontally, vertically or diagonally) while
preventing the other player from doing so. The implementation allows to play the game against a artificial agent utilizing
a depth limited [MiniMax](https://en.wikipedia.org/wiki/Minimax) search to determine its next move. The search is based on [Negamax](https://en.wikipedia.org/wiki/Negamax)
and uses [Alpha-Beta pruning](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning) to optimize search.
