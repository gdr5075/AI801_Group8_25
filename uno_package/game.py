from uno_package import deck, player, card
import random
import time

class Game:
    def __init__(self, players, hasHuman):
        self.deck = deck.UnoMainDeck()
        self.playPile = []
        self.players = players
        self.winning_player = None
        self.turn_count = 0
        self.isClockwise = True
        self.hasHuman = hasHuman
        ## this is in the case of draw4 or draw2, need a way to tell player
        self.nextPlayerAction = None
        ## decide turn order randomly and set first player
        random.shuffle(players)
        ## setting to negative 1 because of how the gameplay loop is currently
        self.currentPlayer = -1


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

    ##TODO: implement game state to send to player
    def play(self):
        print(f'beginning top card {self.get_top_play_card()}')
        self.get_current_game_state()
        while self.is_game_over() == False and self.turn_count < 2000:
            self.turn_count+=1
            print(f"Turn {self.turn_count}")
            self.get_next_player().play(self, self.get_current_game_state())
            self.handle_special_cards()
            #print(f'current top card {self.get_top_play_card()}')
            if(self.deck.is_empty()):
                self.add_play_pile_to_main_deck()
            if(self.hasHuman):
                time.sleep(1)
        if(self.winning_player != None):
            print(f"{self.winning_player.name} won the game")
        return self.winning_player
        
    def get_next_player(self):
        direction = 1 if self.isClockwise else -1
        self.currentPlayer += direction

        self.handle_player_limits()
        if(self.nextPlayerAction != None):
            match (self.nextPlayerAction):
                case card.VALUE.SKIP:
                    print(f"Skipping {self.players[self.currentPlayer].name}")
                    self.currentPlayer += direction
                    self.nextPlayerAction = None
                case card.VALUE.DRAW2:
                    self.draw_cards(self.players[self.currentPlayer], 2)
                    self.currentPlayer += direction
                    self.nextPlayerAction = None
                case card.VALUE.DRAW4:
                    self.draw_cards(self.players[self.currentPlayer], 4)
                    self.currentPlayer += direction
                    self.nextPlayerAction = None

        self.handle_player_limits()

        return self.players[self.currentPlayer]

    def handle_player_limits(self):
        #Handle Overflow
        if(self.currentPlayer > self.players.__len__()-1):
            self.currentPlayer -= (self.players.__len__())
            print(f"Handled overflow")

        #Handle Underflow
        if(self.currentPlayer < 0):
            self.currentPlayer += (self.players.__len__())
            print(f"Handled under")

    def is_game_over(self):
        for player in self.players:
            if player.card_count() == 0:
                self.winning_player = player
                return True
        return False
    
    ## get game's current state
    ## state = [topCard, [player_hand_counts]]
    def get_current_game_state(self):
        currentState = {
            "topCard": self.get_top_play_card(),
            "handCounts": [],
            "isClockwise": self.isClockwise,
            "turnOrder": []
        }
        for p in self.players:
            hand = {"name": p.name, "count": p.card_count()}
            currentState["handCounts"].append(hand)
            currentState["turnOrder"].append(p.name)
        return currentState

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

        #Handle Reverse
        top = self.get_top_play_card()
        match (top.value):

            case card.VALUE.REVERSE:
                self.isClockwise = not self.isClockwise
                return

            case card.VALUE.SKIP:
                self.nextPlayerAction = card.VALUE.SKIP
                return

            case card.VALUE.DRAW2:
                self.nextPlayerAction = card.VALUE.DRAW2
                return

            case card.VALUE.DRAW4:
                self.nextPlayerAction = card.VALUE.DRAW4
                return

            case _:
                #No Action to take
                return