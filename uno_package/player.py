
class Player:
    def __init__(self, name):
        self.name = name
        self.hand = []

    def add_to_hand(self, cards):
        self.hand += cards

    def show_hand(self):
        print(self.name)
        print(self.hand)

    def draw_card(self):
        pass

    def play_card(self):
        pass

    def call_uno(self):
        pass