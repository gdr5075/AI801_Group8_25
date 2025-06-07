from uno_package import deck, player, card
import random

class Game:
    def __init__(self, players):
        self.deck = deck.UnoMainDeck()
        self.discardPile = []
        self.players = players
        
        ## decide turn order randomly
        random.shuffle(players)

        ## deal initial hands
        for player in players:
            self.deal_cards(player, 7)

        ## get the top card, can't be either wild card
        while True:
            c = self.deck.pop()
            if c.value == card.VALUE.DRAW4 or c.value == card.VALUE.NORMAL:
                self.discardPile.append(c)
            else: 
                self.add_discard_pile_to_main_deck()
                self.discardPile.append(c)
                break

        self.print_status()

    def start(self):
        pass

    def is_game_over(self):
        pass

    def add_discard_pile_to_main_deck(self):
        while self.discardPile.__len__() > 0:
            self.deck.push(self.discardPile.pop())
        self.deck.shuffle()
    
    def print_status(self):
        print(self.deck.size())
        print(self.discardPile.__len__())
        print(self.discardPile[self.discardPile.__len__() - 1])

    def deal_cards(self, player, number):
        cards = []
        for i in range(number):
            ## add cards from discard pile, except top of stack, to main deck and reshuffle 
            if self.deck.is_empty():
                c = self.discardPile.pop()
                self.add_discard_pile_to_main_deck()
                pass
            cards.append(self.deck.pop())
        player.add_to_hand(cards)