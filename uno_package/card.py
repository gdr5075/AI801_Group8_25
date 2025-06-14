from enum import Enum

class COLOR(Enum):
    RED = 'Red'
    GREEN = 'Green'
    YELLOW = 'Yellow'
    BLUE = 'Blue'
    WILD = 'Wild'

class VALUE(Enum):
    ZERO = '0'
    ONE = '1'
    TWO = '2'
    THREE = '3'
    FOUR = '4'
    FIVE = '5'
    SIX = '6'
    SEVEN = '7'
    EIGHT = '8'
    NINE = '9'
    REVERSE = 'Reverse'
    SKIP = 'Skip'
    DRAW2 = 'Draw 2'
    DRAW4 = 'Draw 4'
    NORMAL = 'Normal'

class Card:
    def __init__(self, color, value, points):
        self.color = color
        self.value = value
        self.points = points
    def __repr__(self):
        return f"{self.color.value} | {self.value.value}"
    