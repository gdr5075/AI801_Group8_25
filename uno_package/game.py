from uno_package import deck, player, card
import random

class Game:
    def __init__(self, players):
        self.deck = deck.UnoMainDeck()
        self.playPile = []
        self.players = players
        self.winning_player = None
        self.turn_count = 0
        ## decide turn order randomly
        random.shuffle(players)

        ## deal initial hands
        for player in players:
            self.deal_cards(player, 7)

        ## get the top card, can't be either wild card
        while True:
            c = self.deck.pop()
            if c.value == card.VALUE.DRAW4 or c.value == card.VALUE.NORMAL:
                self.playPile.append(c)
            else: 
                self.add_play_pile_to_main_deck()
                self.playPile.append(c)
                break

        #self.print_status()

    def play(self):
        while self.is_game_over() == False and self.turn_count < 2000:
            self.turn_count+=1
            self.get_next_player().play(self)
            if(self.deck.is_empty):
                self.add_play_pile_to_main_deck()
        if(self.winning_player != None):
            print(f"{self.winning_player.name} won the game")
        return self.winning_player
        
    def get_next_player(self):
        #todo - make reverse and skip cards work
        #probably can be done by using an array instead of a list and just adding and subtracting
        #from an index based on moves + handling wrap around cases
        current_player = self.players.pop(0)
        self.players.append(current_player)
        return current_player

    def is_game_over(self):
        for player in self.players:
            if player.card_count() == 0:
                self.winning_player = player
                return True
        return False

    #todo - make sure this leaves the last card in play
    def add_play_pile_to_main_deck(self):
        while self.playPile.__len__() > 1:
            self.deck.push(self.playPile.pop())
        self.deck.shuffle()
    
    def print_status(self):
        pass
        #print(self.deck.size())
        #print(self.playPile.__len__())
        #print(self.playPile[self.playPile.__len__() - 1])

    def deal_cards(self, player, number):
        cards = []
        for i in range(number):
            ## add cards from discard pile, except top of stack, to main deck and reshuffle 
            if self.deck.is_empty():
                c = self.playPile.pop()
                self.add_play_pile_to_main_deck()
                pass
            cards.append(self.deck.pop())
        player.add_to_hand(cards)


    def get_valid_moves(self, player):
        valid_moves = []
        play_top = self.playPile[0]
        for card in player.hand:
            if card.color == play_top.color or card.value == play_top.value or card.value == "Wild":
                #print(f"Valid card from {player.name} - they have a {card.color} {card.value} and the play top is {play_top.color} {play_top.value}")
                valid_moves.append(card)
        return valid_moves

    def play_card(self, card):
        self.playPile.insert(0, card)

    def draw_card(self, player):
        player.hand.append(self.deck.pop())