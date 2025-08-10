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
    
    ## this gets the next agent in the current direction without changing the current agent
    def get_next_agent(self, direction) -> any:
        """Get the next agent."""
        agentnum = (self._current_agent + direction) % len(self.agent_order)
        return self.agent_order[agentnum - 1]

class UnoRLLibEnv(gym.Env):
    wins = 0
    games = 0
    truncates = 0
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
        
        # """
        # Our AgentSelector utility allows easy cyclic stepping through the agents list.
        # """
        self._agent_selector = UnoAgentSelector(self.agents)
        self.current_player = self._agent_selector.reset()

        ## last 61 are action mask
        self.observation_space = gym.spaces.Dict({"observations": gym.spaces.Box(low=0.0, high=108.0, shape=(75,), dtype=np.float32), 
                                                 "action_mask": gym.spaces.Box(low=0.0, high=1.0, shape=(61,), dtype=np.float32)})
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

        self.deck = deck.UnoMainDeck()
        self.playPile = []
        self.winning_player = None
        self.turn_count = 0
        self.isClockwise = True
        self.wildColor = None 

        self.current_player = self._agent_selector.reset()
        
        self.reward = 0
        self.terminated = False
        self.info = {}
        self.truncated = False

        for p in self.players:
            self.players[p].set_player_count(self.get_player_count())
        
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
        
        obs = self.observe(self.trainingAgent)

        return (
            obs,
            self.info,
        )


    def step(self, action):
        #print(f'turn count: {self.turn_count}')
        #print(f'Step called with action_dict: {action}')
        #print(f'Top card: {self.get_top_play_card()}')
        #print(f'Agent has number of cards: {len(self.players[self.trainingAgent].hand)}')
        #print(f'Action: {action}')
        self.reward = 0
        skip = False
        # if self.turn_count > 500:
        #     self.truncated = True
        #     self.reward += self.reward_values['lose']
        #     UnoRLLibEnv.truncates += 1
        #     return self.observe(self.trainingAgent), self.reward, self.terminated, self.truncated, {}

        # if utils.available_moves_to_action_mask(utils.hand_to_state_rep(self.get_valid_moves_for_player(self.players[self.trainingAgent])))[action] == 0:
        #     self.reward += self.reward_values['invalid_action']
        #     self.turn_count += 1
        #     return self.observe(self.trainingAgent), self.reward, self.terminated, self.truncated, {}


        direction = 1 if self.isClockwise else -1

        #print(f'Current direction: {direction}')

        # gets a tuple of card representation and wild color
        playedCardRepr = utils.action_to_card_rep(action)
        #print(f'Played card representation: {playedCardRepr}')

        # if the agent's action is draw, this will be true if they draw a playable card
        agentDrewPlayableCard = False

        ## player is drawing
        if not playedCardRepr:
            self.draw_card(self.trainingAgent)
            #print(f'{self.trainingAgent} drew a card')
            self.reward += self.reward_values['draw']
            ## if player drew card to play, set the boolean to true so it won't skip to the next player for the next step
            if len(self.get_valid_moves_for_player(self.players[self.trainingAgent])) != 0:
                #print(f'{self.trainingAgent} drew a playable card')
                agentDrewPlayableCard = True
        else:
            #print(self.players[self.trainingAgent].get_hand())
            playedCard = self.players[self.trainingAgent].get_card(playedCardRepr[0])
            #print(f'Played card: {playedCard.color} {playedCard.value}')
            self.play_card(playedCard)
            ## set wild color if wild played
            self.wildColor = playedCardRepr[1] if not None else None
            #print(f'Wild color: {self.wildColor}')
            # check if card does something to next player
            self.handle_rewards(playedCard, direction)
            self.check_auto_action(direction, playedCard)

        direction = 1 if self.isClockwise else -1

        if self.check_win_for_player(self.trainingAgent):
            return self.observe(self.trainingAgent), self.reward, self.terminated, self.truncated, {}

        # if the player drew a playable card, go to next step without setting next player
        if (action != 60 or (action == 60 and not agentDrewPlayableCard)):
            self.turn_count += 1
            self.current_player = self._agent_selector.next(direction)

        ##################################################################################
        while self.current_player != self.trainingAgent:
            _action = self.players[self.current_player].get_action_sa(self.observe(self.current_player), self)
            
            # gets a tuple of card representation and wild color
            playedCardRepr = utils.action_to_card_rep(_action)
            #print(f'Played card representation: {playedCardRepr}')

            # if the agent's action is draw, this will be true if they draw a playable card
            agentDrewPlayableCard = False

            ## player is drawing
            if not playedCardRepr:
                self.draw_card(self.current_player)
                #print(f'{self.current_player} drew a card')
                ## if player drew card to play, set the boolean to true so it won't skip to the next player for the next step
                if len(self.get_valid_moves_for_player(self.players[self.current_player])) != 0:
                    #print(f'{self.current_player} drew a playable card')
                    agentDrewPlayableCard = True
            else:
                #print(self.players[self.current_player].get_hand())
                playedCard = self.players[self.current_player].get_card(playedCardRepr[0])
                #print(f'Played card: {playedCard.color} {playedCard.value}')
                self.play_card(playedCard)
                ## set wild color if wild played
                self.wildColor = playedCardRepr[1] if not None else None
                #print(f'Wild color: {self.wildColor}')
                # check if card does something to next player
                self.check_auto_action(direction, playedCard)

            direction = 1 if self.isClockwise else -1

            if self.check_win_for_player(self.current_player):
                self.reward = np.clip(self.reward, -1, 1)
                return self.observe(self.trainingAgent), self.reward, self.terminated, self.truncated, {}
            
            # if the player drew a playable card, go to next step without setting next player
            if (action != 60 or (action == 60 and not agentDrewPlayableCard)):
                self.turn_count += 1
                self.current_player = self._agent_selector.next(direction)    
        ##################################################################################
        

        return self.observe(self.trainingAgent), self.reward, self.terminated, self.truncated, {}


    def observe(self, agent):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agents' hands, played cards, top_card, clockwise
        """
        #print(f'Observing agent: {agent}')
        obs = {
            "observations": {
            }
        }
        obsSpace = utils.hand_to_state_rep(self.players[agent].hand)
        rowToAdd = np.zeros((15), dtype=float)
        rowToAdd[0] = utils.card_to_action_number(self.get_top_play_card(), self.wildColor)
        rowToAdd[1] = utils.color_to_number(self.wildColor)
        rowToAdd[2] = 0 if self.isClockwise else 1
        cardCounts = [self.players[p].card_count() for p in self._agent_selector.get_agent_list(1)]
        for i in range(len(self.agents)):
            rowToAdd[i+3] = cardCounts[i]
        obs['observations'] = np.vstack((obsSpace, rowToAdd)).flatten().astype(np.float32)
        obs['action_mask'] = utils.available_moves_to_action_mask(utils.hand_to_state_rep(self.get_valid_moves_for_player(self.players[agent])))
        # action_mask = utils.available_moves_to_action_mask(utils.hand_to_state_rep(self.get_valid_moves_for_player(self.players[agent])))
        # fullObs = np.concatenate((fullObs, action_mask))
        # fullObs = fullObs.astype(np.float32)
        return obs


    def handle_rewards(self, playedCard, direction):
        """Handle rewards for the played card."""

        nextAgent = self.players[self._agent_selector.get_next_agent(direction)]
        agentAfterNext = self.players[self._agent_selector.get_next_agent(direction)]
        currentAgent = self.players[self.current_player]
        normalCardCount = currentAgent.normal_playable_card_count(self.get_valid_moves_for_player(currentAgent))

        if normalCardCount > 0 and nextAgent.card_count() > 1:
            self.reward += self.reward_values['bad_special_play']

        if currentAgent.card_count() == 1:
            self.reward += self.reward_values['uno']
        
        #Handle when the player plays a card where they pick a color
        if(playedCard.value == card.VALUE.DRAW4 or playedCard.value == card.VALUE.NORMAL):
            count = currentAgent.card_color_count(playedCard.color)
            reward_val = self.reward_values['color_select']
            self.reward += ((reward_val * count) - (1 * reward_val))

        match (playedCard.value):
            case card.VALUE.REVERSE:
                if(nextAgent.card_count() == 1):
                    reward = self.reward_values['reverse_from_uno']
                    self.reward += reward
                if(nextAgent.card_count() < currentAgent.card_count()):
                    self.reward += self.reward_values['reverse_less_cards']
                elif(nextAgent.card_count() > currentAgent.card_count()):
                    self.reward += self.reward_values['reverse_more_cards']
            case card.VALUE.SKIP:
                
                if(nextAgent.card_count() == 1):
                    reward = self.reward_values['skip_uno']
                    self.reward += reward
                if(nextAgent.card_count() < currentAgent.card_count()):
                    self.reward += self.reward_values['skip_less_cards']
                elif(nextAgent.card_count() > currentAgent.card_count()):
                    self.reward += self.reward_values['skip_more_cards']

            case card.VALUE.DRAW2:
                if(nextAgent.card_count() == 1):
                    reward = self.reward_values['draw2_uno']
                    self.reward += reward
                if(nextAgent.card_count() < currentAgent.card_count()):
                    self.reward += self.reward_values['draw2_less_cards']
                elif(nextAgent.card_count() > currentAgent.card_count()):
                    self.reward += self.reward_values['draw2_more_cards']

            case card.VALUE.DRAW4:
                if(nextAgent.card_count() == 1):
                    reward = self.reward_values['draw4_uno']
                    self.reward += reward
                if(nextAgent.card_count() < currentAgent.card_count()):
                    self.reward += self.reward_values['draw4_less_cards']
                elif(nextAgent.card_count() > currentAgent.card_count()):
                    self.reward += self.reward_values['draw4_more_cards']

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
                UnoRLLibEnv.wins += 1
            else:
                self.reward += self.reward_values['lose']
            UnoRLLibEnv.games += 1
            #print(self.winning_player)
            if self.players[self.trainingAgent].card_count() < 7:
                self.reward += self.reward_values['less_than_start']
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
