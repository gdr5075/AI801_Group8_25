from ray.rllib.env.multi_agent_env import MultiAgentEnv
import random
import gymnasium as gym
from gymnasium.utils import seeding
from uno_package import deck, card, utils, player
from pettingzoo.utils import AgentSelector
import numpy as np


class UnoAgentSelector(AgentSelector):
    ## needed to edit because reverse is in uno
    def next(self, direction) -> any:
        """Get the next agent."""
        self._current_agent = (self._current_agent + direction) % len(self.agent_order)
        self.selected_agent = self.agent_order[self._current_agent - 1]
        return self.selected_agent
    
    def reset(self) -> any:
        """Reset to the original order."""
        self.reinit(self.agent_order)
        return self.next(1)
    
    ## gets agents list where first index is current player and the next agents are in game's current direction
    def get_agent_list(self, direction):
        agentNumber = (self._current_agent) % len(self.agent_order)
        agents = []
        for _ in range(len(self.agent_order)):
            agentNumber = (agentNumber) % len(self.agent_order)
            agents.append(self.agent_order[agentNumber - 1])
            agentNumber += direction
        return agents
    
    def randomize(self):
        random.shuffle(self.agent_order)
        return self.next(1)

    ## this gets the next agent in the current direction without changing the current agent
    def get_next_agent(self, direction) -> any:
        """Get the next agent."""
        agentnum = (self._current_agent + direction) % len(self.agent_order)
        return self.agent_order[agentnum - 1]

class PlaygroundEnv(gym.Env):

    def __init__(self, config=None):
        #print('Initializing UnoRLLibEnv')
        #players is a dict of player objects with the key being the player name
        self.players = config.get("players", None)
        self.hasHuman = config.get("hasHuman", False)
        self.reward_values = config.get("reward_values", None)
        super().__init__()

        #active agents
        names = [p for p in self.players.keys()]

        self.trainingAgent = names[0]

        #agents are just the player names
        self.agents = names
        #print(f'Agents: {self.agents}')
        self.randomize = False
        # """
        # Our AgentSelector utility allows easy cyclic stepping through the agents list.
        # """
        self._agent_selector = UnoAgentSelector(self.agents)
        self.current_player = self._agent_selector.reset()

        ## last 61 are action mask
        self.observation_space = gym.spaces.Box(low=0, high=108, shape=(136,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(61)

        self.deck = deck.UnoMainDeck()
        self.playPile = []
        self.winning_player = None
        self.turn_count = 0
        self.isClockwise = True
        self.wildColor = None

    def reset(self, *, seed=None, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        ##print('Resetting UnoRLLibEnv')
        super().reset(seed=seed, options=options)

        for p in self.players:
            self.players[p].set_player_count(self.get_player_count())

        self.deck = deck.UnoMainDeck()
        self.playPile = []
        self.winning_player = None
        self.turn_count = 0
        self.isClockwise = True
        self.wildColor = None 

        self.current_player = self._agent_selector.reset()

        if(self.randomize):
            self.current_player = self._agent_selector.randomize()
        
        self.reward = 0
        self.terminated = False
        self.info = {}
        self.truncated = False
        
        ## deal initial hands
        for player in self.agents:
            self.players[player].clear_hand()
            self.deal_cards(self.players[player], 7)

        ## get the top card, can't be either wild card
        while True:
            c = self.deck.pop()
            if c.color == card.COLOR.WILD:
                self.playPile.append(c)
            else: 
                self.add_play_pile_to_main_deck()
                self.playPile.append(c)
                break
        
        obs = self.observe(self.current_player)

        return (
            obs,
            self.info,
        )


    def step(self, action):
        #print(f'Step called with action {action}')
        direction = 1 if self.isClockwise else -1

        # gets a tuple of card representation and wild color
        playedCardRepr = utils.action_to_card_rep(action)
        #print(f'This makes the played representation {playedCardRepr}')

        # if the agent's action is draw, this will be true if they draw a playable card
        agentDrewPlayableCard = False
        
        if not playedCardRepr:
            ## player is drawing
            self.draw_card(self.current_player)
            #print(f'{self.trainingAgent} drew a card')
            ## if player drew card to play, set the boolean to true so it won't skip to the next player for the next step
            if len(self.get_valid_moves_for_player(self.players[self.current_player])) != 0:
                #print(f'{self.trainingAgent} drew a playable card')
                agentDrewPlayableCard = True
        else:
            #print(self.players[self.trainingAgent].get_hand())
            playedCard = self.players[self.current_player].get_card(playedCardRepr[0])
            #print(f'Played card: {playedCard.color} {playedCard.value}')
            if not playedCard:
                #Let the agent try again
                self.reward = self.reward - 1
                #print(f"Invalid action: {action} by {self.current_player}")
                self.draw_card(self.current_player)
                self.current_player = self._agent_selector.next(direction)
                return self.observe(self.current_player), self.reward, self.terminated, self.truncated, {}
            self.play_card(playedCard)
            ## set wild color if wild played
            self.wildColor = playedCardRepr[1] if not None else None
            #print(f'Wild color: {self.wildColor}')
            # check if card does something to next player
            self.handle_rewards(playedCard, direction)
            self.check_auto_action(direction, playedCard)

        direction = 1 if self.isClockwise else -1

        if self.check_win_for_player(self.current_player):
            return self.observe(self.current_player), self.reward, self.terminated, self.truncated, {}

        # if the player drew a playable card, go to next step without setting next player
        if (action != 60 or (action == 60 and not agentDrewPlayableCard)):
            self.turn_count += 1
            self.current_player = self._agent_selector.next(direction)

        return self.observe(self.current_player), self.reward, self.terminated, self.truncated, {}


    def observe(self, agent):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agents' hands, played cards, top_card, clockwise
        """
        #print(f'Observing agent: {agent}')
        obsSpace = {}
  
        obsSpace[agent] = utils.hand_to_state_rep(self.players[agent].hand)
        rowToAdd = np.zeros((15), dtype=float)
        rowToAdd[0] = utils.card_to_action_number(self.get_top_play_card(), self.wildColor)
        rowToAdd[1] = utils.color_to_number(self.wildColor)
        rowToAdd[2] = 0 if self.isClockwise else 1
        cardCounts = [self.players[p].card_count() for p in self._agent_selector.get_agent_list(1)]
        for i in range(len(self.agents)):
            rowToAdd[i+3] = cardCounts[i]
        fullObs = np.vstack((obsSpace[agent], rowToAdd)).flatten()
        action_mask = utils.available_moves_to_action_mask(utils.hand_to_state_rep(self.get_valid_moves_for_player(self.players[agent])))
        fullObs = np.concatenate((fullObs, action_mask))
        fullObs = fullObs.astype(np.float32)
        return fullObs


    def handle_rewards(self, playedCard, direction):
        """Handle rewards for the played card."""

        nextAgent = self.players[self._agent_selector.get_next_agent(direction)]
        
        match (playedCard.value):
            case card.VALUE.REVERSE:
                if(nextAgent.card_count() == 1):
                    reward = self.reward_values['reverse_from_uno']
                    self.reward += reward
                    #print(f"{self.current_player} gets reward {reward} for reversing away from {nextAgent.name} with 1 card")
                return
            case card.VALUE.SKIP:
                if(nextAgent.card_count() == 1):
                    reward = self.reward_values['skip_uno']
                    self.reward += reward
                    #print(f"{self.current_player} gets reward {reward} for skipping {nextAgent.name} with 1 card")
                return

            case card.VALUE.DRAW2:
                if(nextAgent.card_count() == 1):
                    reward = self.reward_values['draw2_uno']
                    self.reward += reward
                    #print(f"{self.current_player} gets reward {reward} for making {nextAgent.name} draw 2 with 1 card")
                return

            case card.VALUE.DRAW4:
                if(nextAgent.card_count() == 1):
                    reward = self.reward_values['draw4_uno']
                    self.reward += reward
                    #print(f"{self.current_player} gets reward {reward} for making {nextAgent.name} draw 4 with 1 card")
                return
        
        self.reward += self.reward_values['turn']

    ## checks if special action happens to the next player
    ## if it happens to a player, it will perform the action and/or skip their turn
    def check_auto_action(self, direction, playedCard):
         # if the top card is no longer a wild card, reset the chosen color
        if(not self.get_top_play_card().color == card.COLOR.WILD):
            self.wildColor = None
        
        match (playedCard.value):
            case card.VALUE.REVERSE:
                self.isClockwise = not self.isClockwise
                #print(f"Reversing turn order")
                return
            case card.VALUE.SKIP:
                self._agent_selector.next(direction)
                #print(f"Skipping {self._agent_selector.selected_agent}")
                return

            case card.VALUE.DRAW2:
                self._agent_selector.next(direction)
                self.draw_cards(self._agent_selector.selected_agent, 2)
                #print(f"{self._agent_selector.selected_agent} drawing 2 cards")
                return

            case card.VALUE.DRAW4:
                self._agent_selector.next(direction)
                self.draw_cards(self._agent_selector.selected_agent, 4)
                #print(f"{self._agent_selector.selected_agent} drawing 4 cards")
                return
    
    def check_win_for_player(self, player) -> bool:
        ## if player's hand is empty, they win
        if len(self.players[player].hand) == 0:
            self.terminated = True
            self.winning_player = player
            if self.winning_player == self.trainingAgent:
                self.reward += self.reward_values['win']
            else:
                self.reward += self.reward_values['lose']
            return True
        return False
            

    def get_turn_order(self):
        return [p.name for p in self.players]
    
    def get_turn_direction(self):
        return 'clockwise' if self.isClockwise else 'counterclockwise'
    
    def get_hand_counts(self):
        #return [{"name": p.name, "count": p.card_count()} for p in self.players]
        agentList = self._agent_selector.get_agent_list(1)
        cardCounts = [{p : self.players[p].card_count()} for p in agentList]
        return cardCounts
    
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
    
    def get_valid_moves_for_player(self, player):
        valid_moves = []
        play_top = self.get_top_play_card()
        for pCard in player.get_hand():
            if pCard.color == play_top.color or pCard.value == play_top.value or pCard.color == card.COLOR.WILD or pCard.color.value == self.wildColor:
                valid_moves.append(pCard)
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
        self.players[player].get_hand().append(self.draw_card_from_deck())


    ## draws multiple cards from the deck
    ## useful for draw4 and draw2
    def draw_cards(self, player, number):
        for _ in range(number):
            self.draw_card(player)

    def shuffle_players(self):
        random.shuffle(self.players)
    
    
    def get_player(self, player):
        return self.players[player]
    
    def set_randomize(self, randomize):
        self.randomize = randomize