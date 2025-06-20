import random

class Player:
    def __init__(self, name):
        self.name = name
        self.hand = []

    def play(self, game):
        #print(f"Player {self.name} is currently playing")
        self.show_hand()
        moves = game.get_valid_moves(self)
        if(len(moves) == 0):
            game.draw_card(self)
        else:
            idx = random.choice(moves)
            cardToPlay = self.hand.pop(idx)
            #print(f"player {self.name} playing {cardToPlay}")
            game.play_card(cardToPlay)


    def add_to_hand(self, cards):
        self.hand += cards

    def show_hand(self):
        pass
        #print(self.name)
        #print(self.hand)

    def draw_card(self):
        pass

    def play_card(self):
        pass

    def call_uno(self):
        pass

    def card_count(self):
        return self.hand.__len__()

    def hand(self):
        return self.hand