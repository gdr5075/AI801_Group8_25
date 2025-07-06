from uno_package import deck, player, card, utils
import random
import time
import gymnasium as gym
import numpy as np

class GameEnv(gym.Env):
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
        ## setting to negative 1 because of how the gameplay loop is currently
        self.currentPlayer = -1
        self.wildColor = None

        ## deal initial hands
        for player in players:
            self.deal_cards(player, 7)

        ## get the top card, can't be either wild card
        while True:
            c = self.deck.pop()
            if c.color == card.COLOR.WILD:
                self.playPile.append(c)
            else: 
                self.add_play_pile_to_main_deck()
                self.playPile.append(c)
                break

        ##for gym
        obsSpace = {}
        for player in players:
            obsSpace[player.name] = gym.spaces.MultiDiscrete([5,15], dtype=int)
        obsSpace['played_cards'] = gym.spaces.MultiDiscrete([5,15], dtype=int)
        obsSpace["top_card"] = gym.spaces.Text(self.get_top_play_card().__repr__())
        obsSpace["clockwise"] = gym.spaces.Text("clockwise")
        #obsSpace["hand_counts"] = gym.spaces.MultiDiscrete([len(self.players)], dtype=int)
        
        self.observation_space = gym.spaces.Dict(obsSpace)
        
        ## rgby 0-9 skip, reverse, d2, wild, d4
        self.action_space = gym.spaces.Sequence(61)

    # Convert internal state to observation format.  
    # returns dict with each player hand, played cards, top card, and current direction of play 
    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        obsSpace = {}
        for player in self.players:
            obsSpace[player.name] = utils.hand_to_dict(player.hand)
        obsSpace['played_cards'] = gym.spaces.MultiDiscrete([5,15], dtype=int)
        obsSpace["top_card"] = gym.spaces.Text(self.get_top_play_card().__repr__())
        obsSpace["clockwise"] = gym.spaces.Text("clockwise")
        return obsSpace
    
    ## TODO implement for debugging purposes https://gymnasium.farama.org/introduction/create_custom_env/
    def _get_info(self):
        return {}
    
    ## used to start a new episode
    ## an episode in our case is a game
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # IMPORTANT: Must call this first to seed the random number generator
        #super().reset(seed=seed)

        self.deck = deck.UnoMainDeck()
        self.playPile = []
        self.winning_player = None
        self.turn_count = 0
        self.isClockwise = True
        self.nextPlayerAction = None
        self.currentPlayer = -1
        self.wildColor = None

        #TODO: may want to test shuffling and not shuffling to see if going first makes a difference in agent's win %

        ## deal initial hands
        for player in self.players:
            player.clear_hand()
            self.deal_cards(player, 7)

        ## get the top card, can't be either wild card
        while True:
            c = self.deck.pop()
            if c.color == card.COLOR.WILD:
                self.playPile.append(c)
            else: 
                self.add_play_pile_to_main_deck()
                self.playPile.append(c)
                break

        return self._get_obs(), self._get_info() 
    
    ## contains core env logic. It takes an action and updates the env state and returns results
    ## args: action which is 0-60
    def step(self, player, action):
        pass

    def play(self):
        print(f'beginning top card {self.get_top_play_card()}')
        while self.is_game_over() == False and self.turn_count < 2000:
            self.turn_count+=1
            print(f"Turn {self.turn_count}")
            next_player = self.get_next_player()
            card_played = self.execute_player_turn(next_player)
            if card_played:
                #only handle special cards if the player actually played
                self.handle_special_cards()
            if(self.deck.is_empty()):
                self.add_play_pile_to_main_deck()
            if(self.hasHuman):
                time.sleep(1)
        if(self.winning_player != None):
            print(f"{self.winning_player.name} won the game")
        return self.winning_player
        
    def execute_player_turn(self, player):
        #We handle special cards after a player plays them, so by the time we get here we're only concerned
        #with what the next valid player is going to do

        #1 - Check play options
        moves = self.get_valid_moves(player)
        if(len(moves) == 0):
            print(f"{player.name} has no valid moves and has to draw: {player.get_hand()}")
            self.draw_card(player)
            print(f" {player.name} drew so now their hand is: {player.get_hand()}")
            moves = self.get_valid_moves(player) #Refresh moves

        #If player has options, let them move
        if(len(moves) > 0):
            player.play(self)
            return True

        return False


    def get_next_player(self):
        direction = 1 if self.isClockwise else -1
        self.currentPlayer += direction

        self.handle_player_limits()
        # if the top card is no longer a wild card, reset the chosen color
        if(not self.get_top_play_card().color == card.COLOR.WILD):
            self.wildColor = None
        
        if(self.nextPlayerAction != None):
            match (self.nextPlayerAction):
                case card.VALUE.SKIP:
                    print(f"Skipping {self.players[self.currentPlayer].name}")
                    self.currentPlayer += direction
                    self.nextPlayerAction = None

                case card.VALUE.DRAW2:
                    print(f"{self.players[self.currentPlayer].name} drawing 2 cards")
                    self.draw_cards(self.players[self.currentPlayer], 2)
                    self.currentPlayer += direction
                    self.nextPlayerAction = None

                case card.VALUE.DRAW4:
                    print(f"{self.players[self.currentPlayer].name} drawing 4 cards")
                    self.draw_cards(self.players[self.currentPlayer], 4)
                    self.currentPlayer += direction
                    self.nextPlayerAction = None

        self.handle_player_limits()

        return self.players[self.currentPlayer]

    def handle_player_limits(self):
        #Handle Overflow
        if(self.currentPlayer > self.players.__len__()-1):
            self.currentPlayer -= (self.players.__len__())
            #print(f"Handled overflow")

        #Handle Underflow
        if(self.currentPlayer < 0):
            self.currentPlayer += (self.players.__len__())
            #print(f"Handled under")

    def is_game_over(self):
        for player in self.players:
            if player.card_count() == 0:
                self.winning_player = player
                return True
        return False
    
    def get_turn_order(self):
        return [p.name for p in self.players]
    
    def get_turn_direction(self):
        return 'clockwise' if self.isClockwise else 'counterclockwise'
    
    def get_hand_counts(self):
        return [{"name": p.name, "count": p.card_count()} for p in self.players]
    
    def get_chosen_wild_color(self):
        return self.wildColor
    
    def get_player_count(self):
        return len(self.players)
     
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

        for i in range(len(player.get_hand())):
            pCard = player.get_hand()[i]
            if pCard.color == play_top.color or pCard.value == play_top.value or pCard.color == card.COLOR.WILD or pCard.color == self.wildColor:
                #print(f"Valid card from {player.name} - they have a {pCard.color} {pCard.value} and the play top is {play_top.color} {play_top.value}")
                valid_moves.append(i)
        return valid_moves
    
    def get_top_play_card(self):
        return self.playPile[self.playPile.__len__()-1]

    def play_card(self, play_card):
        self.playPile.append(play_card)

    def choose_wild_color(self, color):
        self.wildColor = color

    ## gets a card from the deck
    ## if empty after, call method to shuffle playpile back into deck
    def draw_card_from_deck(self):
        c = self.deck.pop()
        if(self.deck.is_empty()):
            self.add_play_pile_to_main_deck()
        return c

    ## draws a single card from the deck
    def draw_card(self, player):
        player.get_hand().append(self.draw_card_from_deck())


    ## draws multiple cards from the deck
    ## useful for draw4 and draw2
    def draw_cards(self, player, number):
        for i in range(number):
            self.draw_card(player)


    def shuffle_players(self):
        random.shuffle(self.players)

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