import numpy as np
from uno_package import game
                        ## 0-9, skip, reverse, d2, wild, d4
cardMatrix = np.zeros((4,4,15), dtype=int) 

## assign wilds to a color to simplify state to 4 rows 8/4 = 2 wilds of each color
## each color can have at most 2 cards of a type
# 4 rows
# 15 columms
# 4 players + 1 draw pile
# 4 x 15 x 5 = 300
## 2^300 states 

## 61 actions

class Env():
    def __init__(self, game):
        self.playerCount = game.get_player_count()
        self.stateShape = [[4,4,15] for _ in range(self.playerCount)]
        self.actionShape = [None for _ in range(self.playerCount)]