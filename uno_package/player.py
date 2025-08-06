import random
from uno_package import card, utils

class Player:
    def __init__(self, name):
        self.name = name
        self.hand = []

    def play(self, game):
        #print(f"Player {self.name} is currently playing")
        
        #self.show_hand()
        moves = game.get_valid_moves(self)
        #print(f'Has moves {moves}')

        idx = random.choice(moves)
        cardToPlay = self.hand.pop(idx)
        #print(f"player {self.name} playing {cardToPlay}")
        game.play_card(cardToPlay)
        if(cardToPlay.color == card.COLOR.WILD):
            color = random.choice(utils.normal_color_list)
            game.choose_wild_color(color)
            #print(f"Wild card played, {self.name} chose {utils.colorize_text_by_color_name(color, color)} as the next color")

    def get_action(self, observation):
        moves = utils.state_rep_to_action_numbers_list(observation)
        if len(moves) == 0:
            return utils.card_rep_to_action_number('DRAW')
        action = self.decide(moves)
        (cardToPlay, chosenColor) = utils.action_to_card_rep(action)
        ctpColor = cardToPlay.split(' | ')[0]

        return action
    
    def get_action_sa(self, observation, env):
        moves = utils.state_rep_to_action_numbers_list(utils.hand_to_state_rep(env.get_valid_moves_for_player(self)))
        if len(moves) == 0:
            return utils.card_rep_to_action_number('DRAW')
        action = self.decide(moves)
        (cardToPlay, chosenColor) = utils.action_to_card_rep(action)
        ctpColor = cardToPlay.split(' | ')[0]
        return action
    
    def decide(self, moves):
        return random.choice(moves)

    def clear_hand(self):
        self.hand = []
    
    def add_to_hand(self, cards):
        self.hand += cards

    def show_hand(self):
        pass
        #print(f"{self.name}'s current hand:")
        #print(f'{self.hand}')

    def draw_card(self):
        pass

    def play_card(self):
        pass

    def call_uno(self):
        pass

    def get_card(self, card_rep):
        #print(f'Getting card {card_rep} from hand')
        for i in range(len(self.hand)):
            #print(f'Checking {self.hand[i]}')
            if self.hand[i].__repr__() == card_rep:
                #print(f'Found card {self.hand[i]} at index {i}')
                return self.hand.pop(i)
        return None

    def card_count(self):
        return self.hand.__len__()


    def card_color_count(self, color):
        count = 0
        for c in self.hand:
            if c.color.value == color:
                count += 1
        return count

    def get_hand(self):
        return self.hand

    def set_player_count(self, count):
        self.player_count = count

    def check_valid_action(self, action, validMoves):
        aCard = utils.action_to_card_rep(action)
        cColor , value = aCard[0].split(' | ')
        for move in validMoves:
            moveCardColor = move.color.value
            moveCardValue = move.value.value
            if((moveCardColor == cColor and moveCardValue == value) or moveCardValue == card.VALUE.NORMAL.value or moveCardValue == card.VALUE.DRAW4.value):
                return True
        return False
class HumanPlayer(Player):
    def __init__(self, name):
        super().__init__(name)

    def play(self, game):

        ##just #printing info for player
        #print(f'{utils.TextCode.RED.value}----------Your turn {self.name}----------{utils.TextCode.RESET.value}')
        
        #print(f'Current top card is {utils.colorize_text_based_on_card_color(game.get_top_play_card(), game.get_top_play_card())}')
        #if(game.get_top_play_card().color == card.COLOR.WILD):
           #print(f'Chosen color is {utils.colorize_text_by_color_name(game.get_chosen_wild_color(), game.get_chosen_wild_color())}')

        #print(f'Turn direction: {game.get_turn_direction()}')
        #print(f'Turn Order: {game.get_turn_order()}')
        #print(f'Hand Counts {game.get_hand_counts()}')
        
        ## tell player current hand
        handStr = ''
        for i in range(len(self.hand)):
            handStr += f"{str(i)}: {utils.colorize_text_based_on_card_color(f'{self.hand[i].__repr__()}', self.hand[i])} "
        #print(f'Current hand: {handStr}')


        moves = game.get_valid_moves(self)
        ##player choosing move
        while(True):
            moveStr = 'Valid moves:'
            moveStrAppend = ''
            for move in moves:
                moveStrAppend += f" {move}: {utils.colorize_text_based_on_card_color(f'{self.hand[move].__repr__()}', self.hand[move])}"

            #print(f'{moveStr + moveStrAppend}')
            ##make sure input is integer and a valid move
            cardToPlay = None
            try:
                choice = int(input("Select number from valid choices above: "))
                #print(f"You chose to play {utils.colorize_text_based_on_card_color(f'{self.hand[choice].__repr__()}', self.hand[choice])}")
                if(not choice in moves):
                    raise ValueError()
                cardToPlay = self.hand.pop(choice)
                game.play_card(cardToPlay)
            except ValueError:
                print("Please enter a valid number for move")

            ## if we chose to play a wild card, need to choose a color
            if(cardToPlay.color == card.COLOR.WILD):
                try:
                    colorChoiceStr = ''
                    for i in range(len(utils.normal_color_list)):
                        colorChoiceStr += f" {i}: {utils.colorize_text_by_color_name(utils.normal_color_list[i], utils.normal_color_list[i])}"
                    #print(colorChoiceStr)
                    choice = int(input("Select number from valid choices above to choose color: "))
                    game.choose_wild_color(utils.normal_color_list[choice])
                    #print(f'You chose the color {utils.colorize_text_by_color_name(utils.normal_color_list[choice], utils.normal_color_list[choice])}')
                except ValueError:
                    print("Please enter a valid number for the color")
            break

    def get_action(self, observation):

        ##just #printing info for player
        #print(f'{utils.TextCode.RED.value}----------Your turn {self.name}----------{utils.TextCode.RESET.value}')
        
        #print(f'Current top card is {utils.colorize_text_based_on_card_color(game.get_top_play_card(), game.get_top_play_card())}')
        #if(game.get_top_play_card().color == card.COLOR.WILD):
           #print(f'Chosen color is {utils.colorize_text_by_color_name(game.get_chosen_wild_color(), game.get_chosen_wild_color())}')

        #print(f'Turn direction: {game.get_turn_direction()}')
        #print(f'Turn Order: {game.get_turn_order()}')
        #print(f'Hand Counts {game.get_hand_counts()}')
        
        ## tell player current hand
        handStr = ''
        for i in range(len(self.hand)):
            handStr += f"{str(i)}: {utils.colorize_text_based_on_card_color(f'{self.hand[i].__repr__()}', self.hand[i])} "
        #print(f'Current hand: {handStr}')


        moves = game.get_valid_moves(self)
        ##player choosing move
        while(True):
            moveStr = 'Valid moves:'
            moveStrAppend = ''
            for move in moves:
                moveStrAppend += f" {move}: {utils.colorize_text_based_on_card_color(f'{self.hand[move].__repr__()}', self.hand[move])}"

            #print(f'{moveStr + moveStrAppend}')
            ##make sure input is integer and a valid move
            cardToPlay = None
            try:
                choice = int(input("Select number from valid choices above: "))
                #print(f"You chose to play {utils.colorize_text_based_on_card_color(f'{self.hand[choice].__repr__()}', self.hand[choice])}")
                if(not choice in moves):
                    raise ValueError()
                cardToPlay = self.hand.pop(choice)
                game.play_card(cardToPlay)
            except ValueError:
                print("Please enter a valid number for move")

            ## if we chose to play a wild card, need to choose a color
            if(cardToPlay.color == card.COLOR.WILD):
                try:
                    colorChoiceStr = ''
                    for i in range(len(utils.normal_color_list)):
                        colorChoiceStr += f" {i}: {utils.colorize_text_by_color_name(utils.normal_color_list[i], utils.normal_color_list[i])}"
                    #print(colorChoiceStr)
                    choice = int(input("Select number from valid choices above to choose color: "))
                    game.choose_wild_color(utils.normal_color_list[choice])
                    #print(f'You chose the color {utils.colorize_text_by_color_name(utils.normal_color_list[choice], utils.normal_color_list[choice])}')
                except ValueError:
                    print("Please enter a valid number for the color")
            break


class RuleBasedPlayer(Player):

    def get_action(self, observation):
        moves = utils.state_rep_to_action_numbers_list(observation)

        if len(moves) == 0:
            return utils.card_rep_to_action_number('DRAW')
        
        action = self.decide(moves, observation)

        return action
    
    def get_action_sa(self, observation, env):
        moves = utils.state_rep_to_action_numbers_list(utils.hand_to_state_rep(env.get_valid_moves_for_player(self)))
        if len(moves) == 0:
            return utils.card_rep_to_action_number('DRAW')
        action = self.decide(moves, observation)
        (cardToPlay, chosenColor) = utils.action_to_card_rep(action)
        ctpColor = cardToPlay.split(' | ')[0]
        return action
    
    def decide(self, moves, observation):

        #Rules for Rule Based Agent
        # The rules shall be executed in this order - if the rule isnt possible, its not played
        # If the rule specifies multiple cards to play, they shall be picked in that order

        #1 - If the next player has less cards than the player - play Draw2, Skip
        #2 - If the next player has less than half the cards of the palyer - play Draw4
        #3 - If the previous player has more cards then the next player - play Reverse
        
        #4 - Play Normal Cards
        #5 - Play Draw2, Skip, Reverse
        #6 - Play Wild Cards

        #Notes - Wild Card color shall be chose as the card the player has the most of, else, just different than the top card
        #Notes - In any instance where two cards of the same value but different color can be played, the card with color
        #that matches the color the player has the most of will be played

        prefered = self.highest_color()

        normal_cards = []
        skips = []
        plus2 = []
        plus4 = []
        reverse = []
        wild = []
        for action in moves:
            aCard = utils.action_to_card_rep(action)
            _ , value = aCard[0].split(' | ')
            if(value == card.VALUE.SKIP.value):
                skips.append(action)
            elif(value == card.VALUE.DRAW2.value):
                plus2.append(action)
            elif(value == card.VALUE.DRAW4.value):
                plus4.append(action)
            elif(value == card.VALUE.REVERSE.value):
                reverse.append(action)
            elif(value == card.VALUE.NORMAL.value):
                wild.append(action)
            else:
                normal_cards.append(action)
            
        # print(normal_cards)
        # print(skips)
        # print(plus2)
        # print(plus4)
        # print(reverse)
        # print(wild)

        topCard = observation[60]
        card_countPlayer = self.card_count()
        card_countNext = observation[64]
        card_countPrevious = observation[63 + self.player_count - 1]

        #Rule 1
        if(card_countNext < card_countPlayer and len(plus2) > 0):
            return self.pick_color(prefered, plus2)
        
        if(card_countNext < card_countPlayer and len(skips) > 0):
            return self.pick_color(prefered, skips)

        #Rule 2
        if(card_countNext < (card_countPlayer/2) and len(plus4) > 0):
            return self.pick_color(prefered, plus4)

        
        #Rule 3
        if(card_countNext < card_countPrevious and len(reverse) > 0):
            return self.pick_color(prefered, reverse)
        
        #Rule 4
        if(len(normal_cards) > 0):
            return self.pick_color(prefered, normal_cards)
        
        #Rule 5
        if(len(skips) > 0):
            return self.pick_color(prefered, skips)
        if(len(plus2) > 0):
            return self.pick_color(prefered, plus2)
        if(len(reverse) > 0):
            return self.pick_color(prefered, reverse)
        
        #Rule 6
        if(len(wild) > 0):
            return self.pick_color(prefered, wild)
        if(len(plus4) > 0):
            return self.pick_color(prefered, plus4)
        
        #Catch all
        return moves[0]
    
    def pick_color(self, highest, actions):
        for a in actions:
            aCard = utils.action_to_card_rep(a)
            cColor , value = aCard[0].split(' | ')
            if(cColor == highest.value):
                return a
            
        return actions[0]


    def highest_color(self):
        color = card.COLOR.BLUE
        red = 0
        green = 0
        blue = 0
        yellow = 0
        for c in self.hand:
            match(c.color):
                case card.COLOR.RED:
                    red+=1
                case card.COLOR.BLUE:
                    blue+=1
                case card.COLOR.GREEN:
                    green+=1
                case card.COLOR.YELLOW:
                    yellow+=1
        if green > blue:
            color = card.COLOR.GREEN
        if yellow > green:
            color = card.COLOR.YELLOW
        if red > yellow:
            color = card.COLOR.RED
        
        return color
        



