import functools, time, random
from typing import Any, Optional
import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding
from uno_package import deck, card, utils, player
from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector, wrappers

class UnoAgentSelector(AgentSelector):
    ## needed to edit because reverse is in uno
    def next(self, direction) -> Any:
        """Get the next agent."""
        self._current_agent = (self._current_agent + direction) % len(self.agent_order)
        self.selected_agent = self.agent_order[self._current_agent - 1]
        return self.selected_agent
    
    def reset(self) -> Any:
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

def env(**kwargs):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env = raw_env(**kwargs)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class raw_env(AECEnv):
    """
    The metadata holds environment constants. From gymnasium, we inherit the "render_modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """
    metadata = {
        "name": "uno_v0",
        "is_parallelizable": False,
        "render_modes": ["human"]
    }

    def __init__(self, players, hasHuman, render_mode=None):
        self.hasHuman = hasHuman

        ## petting zoo variables for AECenv
        self.render_mode = render_mode
        #active agents
        self.agents = players
        #all agents
        self.possible_agents = self.agents[:]

        """
        Our AgentSelector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = UnoAgentSelector(self.agents)
        self._agent_selector.next(1)

        ##for gym/petting zoo
        obsSpaces = {}
        actSpaces = {}
        for player in self.agents:
            obsSpace = {}
            obsSpace[player] = gym.spaces.MultiDiscrete([4,15], dtype=int)
            obsSpace['played_cards'] = gym.spaces.MultiDiscrete([4,15], dtype=int)
            obsSpace['top_card'] = gym.spaces.Text(25)
            obsSpace['chosen_color'] = gym.spaces.Text(6)
            obsSpace['available_moves'] = gym.spaces.MultiDiscrete([4,15], dtype=int)
            obsSpace['clockwise'] = gym.spaces.Discrete(2)
            obsSpace['hand_counts'] = gym.spaces.MultiDiscrete([1,4])
            obsSpaces[player] = gym.spaces.Dict(obsSpace)

            actSpaces[player] = gym.spaces.Discrete(61)
        
        
        self.observation_spaces = gym.spaces.Dict(obsSpaces)
        
        ## rgby 0-9 skip, reverse, d2
        ## w normal d4
        ## draw
        self.action_spaces = gym.spaces.Dict(actSpaces)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
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
        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)
        self.deck = deck.UnoMainDeck()
        self.playPile = []
        self.winning_player = None
        self.turn_count = 0
        self.isClockwise = True
        self.nextPlayerAction = None
        self.wildColor = None

        ## for pettingzoo
        ##reset player order
        self._agent_selector.reset()
        self.agent_selection = self._agent_selector.selected_agent
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}

        # Unlike gymnasium's Env, the environment is responsible for setting the random seed explicitly.
        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        
        ## deal initial hands
        for player in self.agents:
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
    
    def observe(self, agent):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agents' hands, played cards, top_card, clockwise
        """
        obsSpace = {}
  
        obsSpace[agent] = utils.hand_to_state_rep(agent.hand)
        obsSpace['played_cards'] = utils.hand_to_state_rep(self.playPile)
        obsSpace['top_card'] = self.get_top_play_card().__repr__()
        obsSpace['chosen_color'] = self.wildColor if self.wildColor else None
        obsSpace['available_moves'] = utils.hand_to_state_rep(self.get_valid_moves_for_player(agent))
        obsSpace['direction'] = 0 if self.isClockwise else 1
        obsSpace['hand_counts'] = [p.card_count() for p in self._agent_selector.get_agent_list(1 if self.isClockwise else -1)]
        return { 'observation': obsSpace}

    
    ## contains core env logic. It takes an action and updates the env state and returns results
    ## args: action which is 0-54
    def step(self, action):

        stepAgent = self.agent_selection
        
        direction = 1 if self.isClockwise else -1

        # gets a tuple of card representation and wild color
        playedCardRepr = utils.action_to_card_rep(action)
        
        ## player is drawing
        if not playedCardRepr:
            self.draw_card(stepAgent)

            ## if player drew card to play just return so that we don't move to next player. May want to add small negative reward for drawing here still
            if len(self.get_valid_moves_for_player(stepAgent)) != 0:
                return
        else:
            playedCard = stepAgent.get_card(playedCardRepr[0])
            self.play_card(playedCard)
            ##set wild color if wild played
            self.wildColor = playedCardRepr[1] if not None else None
            #check if card does something to next player
            self.check_auto_action(direction, playedCard)

        direction = 1 if self.isClockwise else -1

        ## if players hand is empty, they win
        if len(stepAgent.hand) == 0:
            self.winning_player = stepAgent
            self.terminations = {agent: True for agent in self.agents}

        if self.terminations[stepAgent]:
            for agent in self.agents:
                if self.winning_player == agent:
                    self.rewards[agent] = 1
                else:
                    self.rewards[agent] = -1

        self.rewards[stepAgent] = .01

        self._accumulate_rewards()
        #eventually want to have more rewards, maybe causing player with less cards to gain cards, especially if it is one card 
        #possible rewards, skipping next agent if they have 1 card, reverse away from next agent if they have 1 card, making the agent with less card draw
        self.turn_count += 1
        self.agent_selection = self._agent_selector.next(direction)

    ## checks if special action happens to the next player
    ## if it happens to a player, it will perform the action and/or skip their turn
    def check_auto_action(self, direction, playedCard):
         # if the top card is no longer a wild card, reset the chosen color
        if(not self.get_top_play_card().color == card.COLOR.WILD):
            self.wildColor = None
        
        match (playedCard.value):
            case card.VALUE.REVERSE:
                self.isClockwise = not self.isClockwise
                print(f"Reversing turn order")
                return
            case card.VALUE.SKIP:
                self._agent_selector.next(direction)
                print(f"Skipping {self._agent_selector.selected_agent.name}")
                return

            case card.VALUE.DRAW2:
                self._agent_selector.next(direction)
                self.draw_cards(self._agent_selector.selected_agent, 2)
                print(f"{self._agent_selector.selected_agent.name} drawing 2 cards")
                return

            case card.VALUE.DRAW4:
                self._agent_selector.next(direction)
                self.draw_cards(self._agent_selector.selected_agent, 4)
                print(f"{self._agent_selector.selected_agent.name} drawing 4 cards")
                return
    
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
        player.get_hand().append(self.draw_card_from_deck())


    ## draws multiple cards from the deck
    ## useful for draw4 and draw2
    def draw_cards(self, player, number):
        for _ in range(number):
            self.draw_card(player)

    def shuffle_players(self):
        random.shuffle(self.players)


    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]