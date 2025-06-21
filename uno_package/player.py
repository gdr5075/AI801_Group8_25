import random
from uno_package import card
from enum import Enum

class Player:
    def __init__(self, name):
        self.name = name
        self.hand = []

    def play(self, game, specialAction):
        print(f"Player {self.name} is currently playing")
        if(specialAction == card.VALUE.DRAW2):
            print(f"player {self.name} drawing 2 cards")
            game.draw_cards(self, 2)
            return
        elif(specialAction == card.VALUE.DRAW4):
            print(f"player {self.name} drawing 4 cards")
            game.draw_cards(self, 4)
            return
        
        #self.show_hand()
        moves = game.get_valid_moves(self)
        if(len(moves) == 0):
            print(f"player {self.name} is drawing a card")
            game.draw_card(self)
        else:
            idx = random.choice(moves)
            cardToPlay = self.hand.pop(idx)
            print(f"player {self.name} playing {cardToPlay}")
            game.play_card(cardToPlay)


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

    def hand(self):
        return self.hand
    
class TextCode(Enum):
    RED = "\033[38;5;9m"
    GREEN = "\033[38;5;76m"
    BLUE = "\033[38;5;27m"
    YELLOW = "\033[38;5;226m"
    GRAY = "\033[38;5;249m"
    RESET = "\033[0m"
    
class HumanPlayer(Player):
    def __init__(self, name):
        super().__init__(name)

    ## TODO: show player hand counts to human player
    def play(self, game, specialAction):
        print(f'{TextCode.RED.value}Your turn {self.name}{TextCode.RESET.value}')
        print(f'Current top card is {self.colorize_text_based_on_card_color(game.get_top_play_card(), game.get_top_play_card())}')
        if(specialAction == card.VALUE.DRAW2):
            print(f"player {self.name} drawing 2 cards")
            game.draw_cards(self, 2)
            print(f"New Hand {self.hand}")
            return
        elif(specialAction == card.VALUE.DRAW4):
            print(f"player {self.name} drawing 4 cards")
            game.draw_cards(self, 4)
            print(f"New Hand {self.hand}")
            return
        
        handStr = ''
        for i in range(len(self.hand)):
            handStr += f"{str(i)}: {self.colorize_text_based_on_card_color(f'{self.hand[i].__repr__()}', self.hand[i])} "
        print(f'Current hand: {handStr}')
        moves = game.get_valid_moves(self)
        if(len(moves) == 0):
            print(f"You have no valid moves and have to draw {self.hand}")
            game.draw_card(self)
        else:
           while(True):
                moveStr = 'Valid moves:'
                moveStrAppend = ''
                for move in moves:
                   moveStrAppend += f" {move}: {self.colorize_text_based_on_card_color(f'{self.hand[move].__repr__()}', self.hand[move])}"
                   
                print(f'{moveStr + moveStrAppend}')
                try:
                    choice = int(input("Select number from valid choices above"))
                    print(f'You chose to play {self.hand[choice]}')
                    if(not choice in moves):
                        raise ValueError()
                    cardToPlay = self.hand.pop(choice)
                    game.play_card(cardToPlay)
                    break
                except ValueError:
                    print("Please enter a valid number for move")
        print(self.card_count())

    def colorize_text_based_on_card_color(self, text, c):
        colorText = ''
        if(c.color == card.COLOR.BLUE):
            colorText = f'{TextCode.BLUE.value}{text}{TextCode.RESET.value}'
        elif(c.color == card.COLOR.RED):
            colorText = f'{TextCode.RED.value}{text}{TextCode.RESET.value}'
        elif(c.color == card.COLOR.YELLOW):
            colorText = f'{TextCode.YELLOW.value}{text}{TextCode.RESET.value}'
        elif(c.color == card.COLOR.GREEN):
            colorText = f'{TextCode.GREEN.value}{text}{TextCode.RESET.value}'
        elif(c.color == card.COLOR.WILD):
            colorText = f'{TextCode.GRAY.value}{text}{TextCode.RESET.value}'
        return colorText