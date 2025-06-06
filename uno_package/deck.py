from uno_package.card import Card, COLOR, VALUE
import random

class Deck:
    def __init__(self):
        self.cards = []
        
    def push(self, card):
        self.cards.append(card)

    def pop(self):
        return self.cards.pop()
    
    def is_empty(self):
        return self.cards.__len__() == 0
    
    def peek(self):
        if not self.is_empty():
            return self.cards[-1]
        return None
    
    def size(self):
        return self.cards.__len__()

class UnoMainDeck(Deck):
    def __init__(self):
        super().__init__()
        color_list = [COLOR.BLUE, COLOR.GREEN, COLOR.YELLOW, COLOR.RED]
        for value in VALUE:
            match value:
                case VALUE.ZERO:
                    self.add_cards(color_list, value, int(value.value), 1)
                case VALUE.DRAW4:
                    self.add_cards([COLOR.WILD], value, 50, 4)
                case VALUE.NORMAL:
                    self.add_cards([COLOR.WILD], value, 50, 4)
                case VALUE.DRAW2:
                    self.add_cards(color_list, value, 20, 2)
                case VALUE.REVERSE:
                    self.add_cards(color_list, value, 20, 2)
                case VALUE.SKIP:
                    self.add_cards(color_list, value, 20, 2)
                case _:
                   self.add_cards(color_list, value, int(value.value), 2)
        self.shuffle()

    def add_cards(self, colors, value, points, quantity):
        for color in colors:
            for i in range(quantity):
                self.cards.append(Card(color, value, int(points)))

    def shuffle(self):
        random.shuffle(self.cards)

    def print_all(self):
        for card in self.cards:
            print(card.__repr__())