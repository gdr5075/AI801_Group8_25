import random
from uno_package import card, utils

class Player:
    def __init__(self, name):
        self.name = name
        self.hand = []

    def play(self, game):
        print(f"Player {self.name} is currently playing")
        
        #self.show_hand()
        moves = game.get_valid_moves(self)

        idx = random.choice(moves)
        cardToPlay = self.hand.pop(idx)
        print(f"player {self.name} playing {cardToPlay}")
        game.play_card(cardToPlay)
        if(cardToPlay.color == card.COLOR.WILD):
            color = random.choice(utils.normal_color_list)
            game.choose_wild_color(color)
            print(f"Wild card played, {self.name} chose {utils.colorize_text_by_color_name(color, color)} as the next color")

    def clear_hand(self):
        self.hand = []
    
    def add_to_hand(self, cards):
        self.hand += cards

    def show_hand(self):
        print(f"{self.name}'s current hand:")
        print(f'{self.hand}')

    def draw_card(self):
        pass

    def play_card(self):
        pass

    def call_uno(self):
        pass

    def card_count(self):
        return self.hand.__len__()

    def get_hand(self):
        return self.hand
    
class HumanPlayer(Player):
    def __init__(self, name):
        super().__init__(name)

    def play(self, game):

        ##just printing info for player
        print(f'{utils.TextCode.RED.value}----------Your turn {self.name}----------{utils.TextCode.RESET.value}')
        
        print(f'Current top card is {utils.colorize_text_based_on_card_color(game.get_top_play_card(), game.get_top_play_card())}')
        if(game.get_top_play_card().color == card.COLOR.WILD):
           print(f'Chosen color is {utils.colorize_text_by_color_name(game.get_chosen_wild_color(), game.get_chosen_wild_color())}')

        print(f'Turn direction: {game.get_turn_direction()}')
        print(f'Turn Order: {game.get_turn_order()}')
        print(f'Hand Counts {game.get_hand_counts()}')
        
        ## tell player current hand
        handStr = ''
        for i in range(len(self.hand)):
            handStr += f"{str(i)}: {utils.colorize_text_based_on_card_color(f'{self.hand[i].__repr__()}', self.hand[i])} "
        print(f'Current hand: {handStr}')


        moves = game.get_valid_moves(self)
        ##player choosing move
        while(True):
            moveStr = 'Valid moves:'
            moveStrAppend = ''
            for move in moves:
                moveStrAppend += f" {move}: {utils.colorize_text_based_on_card_color(f'{self.hand[move].__repr__()}', self.hand[move])}"

            print(f'{moveStr + moveStrAppend}')
            ##make sure input is integer and a valid move
            cardToPlay = None
            try:
                choice = int(input("Select number from valid choices above: "))
                print(f"You chose to play {utils.colorize_text_based_on_card_color(f'{self.hand[choice].__repr__()}', self.hand[choice])}")
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
                    print(colorChoiceStr)
                    choice = int(input("Select number from valid choices above to choose color: "))
                    game.choose_wild_color(utils.normal_color_list[choice])
                    print(f'You chose the color {utils.colorize_text_by_color_name(utils.normal_color_list[choice], utils.normal_color_list[choice])}')
                except ValueError:
                    print("Please enter a valid number for the color")
            break