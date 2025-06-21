from uno_package import deck, player, card
import random

class Game:
    def __init__(self, players):
        self.deck = deck.UnoMainDeck()
        self.playPile = []
        self.players = players
        self.winning_player = None
        self.turn_count = 0
        self.isClockwise = True
        self.currentPlayer = 0
        ## this is in the case of draw4 or draw2, need a way to tell player
        self.nextPlayerAction = None
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
        print(f'beginning top card {self.get_top_play_card()}')
        while self.is_game_over() == False and self.turn_count < 2000:
            self.turn_count+=1
            print(f"Turn {self.turn_count}")
            self.get_next_player().play(self, self.nextPlayerAction)
            self.handle_special_cards()
            #print(f'current top card {self.get_top_play_card()}')
            if(self.deck.is_empty()):
                self.add_play_pile_to_main_deck()
        if(self.winning_player != None):
            print(f"{self.winning_player.name} won the game")
        return self.winning_player
        
    def get_next_player(self):
        ## clockwise increment currentPlayer by 1
        if(self.isClockwise):
            if(self.currentPlayer < self.players.__len__() - 1):
                self.currentPlayer += 1
            else:
                self.currentPlayer = 0
        ## counterClockwise decrement currentPlayer by 1
        else:
            if(self.currentPlayer > 0):
                self.currentPlayer -= 1
            else:
                self.currentPlayer = self.players.__len__() - 1

        return self.players[self.currentPlayer]

    def is_game_over(self):
        for player in self.players:
            if player.card_count() == 0:
                self.winning_player = player
                return True
        return False

    #todo - make sure this leaves the last card in play
    def add_play_pile_to_main_deck(self):
        ## pop top card to leave on play pile

        if(self.playPile.__len__() > 0):
            c = self.playPile.pop()

            while self.playPile.__len__() > 0:
                self.deck.push(self.playPile.pop())
            
            self.playPile.append(c)
            self.deck.shuffle()
    
    def print_status(self):
        print(self.deck.size())
        print(self.playPile.__len__())
        print(self.playPile[self.playPile.__len__() - 1])

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
        play_top = self.get_top_play_card()

        for i in range(len(player.hand)):
            pCard = player.hand[i]
            ## TODO: need to handle this better, but for now just go through the loop to handle wild on top
            if play_top.color == card.COLOR.WILD:
                valid_moves.append(i)
                #print(f"Valid card from {player.name} - they have a {pCard.color} {pCard.value} and the play top is {play_top.color} {play_top.value}")
            elif pCard.color == play_top.color or pCard.value == play_top.value or pCard.color == card.COLOR.WILD:
                #print(f"Valid card from {player.name} - they have a {pCard.color} {pCard.value} and the play top is {play_top.color} {play_top.value}")
                valid_moves.append(i)
        return valid_moves
    
    def get_top_play_card(self):
        return self.playPile[self.playPile.__len__()-1]

    def play_card(self, card):
        self.playPile.append(card)

    ## gets a card from the deck
    ## if empty after, call method to shuffle playpile back into deck
    def draw_card_from_deck(self):
        c = self.deck.pop()
        if(self.deck.is_empty()):
            self.add_play_pile_to_main_deck()
        return c

    ## draws a single card from the deck
    def draw_card(self, player):
        player.hand.append(self.draw_card_from_deck())

    ## draws multiple cards from the deck
    ## useful for draw4 and draw2
    def draw_cards(self, player, number):
        for i in range(number):
            self.draw_card(player)

    ## this handles cards in the deck that have special effects
    def handle_special_cards(self):

        if(self.get_top_play_card().value == card.VALUE.REVERSE):
            self.isClockwise = not self.isClockwise
            return
        ## think about changing skip to act like draw2 and draw4 if we want negative reward for being skipped?
        if(self.get_top_play_card().value == card.VALUE.SKIP):
            self.get_next_player()
            return
        if(self.get_top_play_card().value == card.VALUE.DRAW2):
            if(self.nextPlayerAction == None):
                self.nextPlayerAction = card.VALUE.DRAW2
            else:
                self.nextPlayerAction = None
            return
        if(self.get_top_play_card().value == card.VALUE.DRAW4):
            if(self.nextPlayerAction == None):
                self.nextPlayerAction = card.VALUE.DRAW4
            else:
                self.nextPlayerAction = None
            return
