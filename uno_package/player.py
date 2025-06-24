import random
from uno_package import card
from enum import Enum

class Player:
    def __init__(self, name):
        self.name = name
        self.hand = []

    def play(self, game, gameState):
        print(f"Player {self.name} is currently playing")
        
        #self.show_hand()
        moves = game.get_valid_moves(self)

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
    
##ansi codes to color text
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

    def play(self, game, gameState):

        ##just printing info for player
        print(f'{TextCode.RED.value}----------Your turn {self.name}----------{TextCode.RESET.value}')
        print(f'Current top card is {self.colorize_text_based_on_card_color(gameState["topCard"], gameState["topCard"])}')
        print(gameState["isClockwise"])
        if(gameState["isClockwise"]):
            print(f'Turn direction: clockwise')
        else:
            print(f'Turn direction: counterclockwise')
        print(f'Turn Order: {gameState["turnOrder"]}')
        print(f'Hand Counts {gameState["handCounts"]}')
        
        ## tell player current hand
        handStr = ''
        for i in range(len(self.hand)):
            handStr += f"{str(i)}: {self.colorize_text_based_on_card_color(f'{self.hand[i].__repr__()}', self.hand[i])} "
        print(f'Current hand: {handStr}')


        moves = game.get_valid_moves(self)
        ##player choosing move
        while(True):
            moveStr = 'Valid moves:'
            moveStrAppend = ''
            for move in moves:
                moveStrAppend += f" {move}: {self.colorize_text_based_on_card_color(f'{self.hand[move].__repr__()}', self.hand[move])}"

            print(f'{moveStr + moveStrAppend}')
            ##make sure input is integer and a valid move
            try:
                choice = int(input("Select number from valid choices above: "))
                print(f"You chose to play {self.colorize_text_based_on_card_color(f'{self.hand[choice].__repr__()}', self.hand[choice])}")
                if(not choice in moves):
                    raise ValueError()
                cardToPlay = self.hand.pop(choice)
                game.play_card(cardToPlay)
                break
            except ValueError:
                print("Please enter a valid number for move")

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