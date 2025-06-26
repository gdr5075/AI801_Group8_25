from uno_package import card
from enum import Enum

normal_color_list = [card.COLOR.BLUE, card.COLOR.GREEN, card.COLOR.YELLOW, card.COLOR.RED]

##ansi codes to color text
class TextCode(Enum):
    RED = "\033[38;5;9m"
    GREEN = "\033[38;5;76m"
    BLUE = "\033[38;5;27m"
    YELLOW = "\033[38;5;226m"
    GRAY = "\033[38;5;249m"
    RESET = "\033[0m"

def colorize_text_based_on_card_color(text, c):
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

## take a custom ansi color value or value from the enum
def colorize_text(text, color):
     return f'{color}{text}{TextCode.RESET.value}'

def colorize_text_by_color_name(text, color):
    colorText = ''
    if(color == card.COLOR.BLUE):
        colorText = f'{TextCode.BLUE.value}{text}{TextCode.RESET.value}'
    elif(color == card.COLOR.RED):
        colorText = f'{TextCode.RED.value}{text}{TextCode.RESET.value}'
    elif(color == card.COLOR.YELLOW):
        colorText = f'{TextCode.YELLOW.value}{text}{TextCode.RESET.value}'
    elif(color == card.COLOR.GREEN):
        colorText = f'{TextCode.GREEN.value}{text}{TextCode.RESET.value}'
    elif(color == card.COLOR.WILD):
        colorText = f'{TextCode.GRAY.value}{text}{TextCode.RESET.value}'
    return colorText