from uno_package import card
from enum import Enum
import numpy as np

normal_color_list = [card.COLOR.RED, card.COLOR.GREEN, card.COLOR.BLUE, card.COLOR.YELLOW]

COLOR_MAP = {card.COLOR.RED.value: 0, card.COLOR.GREEN.value: 1, card.COLOR.BLUE.value: 2, card.COLOR.YELLOW.value: 3}

# a map of value to its index
VALUE_MAP = {card.VALUE.ZERO.value: 0, card.VALUE.ONE.value: 1, card.VALUE.TWO.value: 2,
            card.VALUE.THREE.value: 3, card.VALUE.FOUR.value: 4, card.VALUE.FIVE.value: 5,
            card.VALUE.SIX.value: 6, card.VALUE.SEVEN.value: 7, card.VALUE.EIGHT.value: 8,
            card.VALUE.NINE.value: 9, card.VALUE.SKIP.value: 10, card.VALUE.REVERSE.value: 11,
            card.VALUE.DRAW2.value: 12, card.VALUE.NORMAL.value: 13, card.VALUE.DRAW4.value: 14}

## 61 actions
## 0-14 RED 0-9, skip, reverse, d2, wild, d4
## 15-29 GREEN 0-9, skip, reverse, d2, wild, d4
## 30-54 BLUE 0-9, skip, reverse, d2, wild, d4
## 55-59 YELLOW 0-9, skip, reverse, d2, wild, d4
## 60 draw
ACTION_MAP = {
    f"{card.COLOR.RED.value} | {card.VALUE.ZERO.value}": 0,
    f"{card.COLOR.RED.value} | {card.VALUE.ONE.value}": 1,
    f"{card.COLOR.RED.value} | {card.VALUE.TWO.value}": 2,
    f"{card.COLOR.RED.value} | {card.VALUE.THREE.value}": 3,
    f"{card.COLOR.RED.value} | {card.VALUE.FOUR.value}": 4,
    f"{card.COLOR.RED.value} | {card.VALUE.FIVE.value}": 5,
    f"{card.COLOR.RED.value} | {card.VALUE.SIX.value}": 6,
    f"{card.COLOR.RED.value} | {card.VALUE.SEVEN.value}": 7,
    f"{card.COLOR.RED.value} | {card.VALUE.EIGHT.value}": 8,
    f"{card.COLOR.RED.value} | {card.VALUE.NINE.value}": 9,
    f"{card.COLOR.RED.value} | {card.VALUE.SKIP.value}": 10,
    f"{card.COLOR.RED.value} | {card.VALUE.REVERSE.value}": 11,
    f"{card.COLOR.RED.value} | {card.VALUE.DRAW2.value}": 12,
    f"{card.COLOR.RED.value} | {card.VALUE.NORMAL.value}": 13,
    f"{card.COLOR.RED.value} | {card.VALUE.DRAW4.value}": 14,
    f"{card.COLOR.GREEN.value} | {card.VALUE.ZERO.value}": 15,
    f"{card.COLOR.GREEN.value} | {card.VALUE.ONE.value}": 16,
    f"{card.COLOR.GREEN.value} | {card.VALUE.TWO.value}": 17,
    f"{card.COLOR.GREEN.value} | {card.VALUE.THREE.value}": 18,
    f"{card.COLOR.GREEN.value} | {card.VALUE.FOUR.value}": 19,
    f"{card.COLOR.GREEN.value} | {card.VALUE.FIVE.value}": 20,
    f"{card.COLOR.GREEN.value} | {card.VALUE.SIX.value}": 21,
    f"{card.COLOR.GREEN.value} | {card.VALUE.SEVEN.value}": 22,
    f"{card.COLOR.GREEN.value} | {card.VALUE.EIGHT.value}": 23,
    f"{card.COLOR.GREEN.value} | {card.VALUE.NINE.value}": 24,
    f"{card.COLOR.GREEN.value} | {card.VALUE.SKIP.value}": 25,
    f"{card.COLOR.GREEN.value} | {card.VALUE.REVERSE.value}": 26,
    f"{card.COLOR.GREEN.value} | {card.VALUE.DRAW2.value}": 27,
    f"{card.COLOR.GREEN.value} | {card.VALUE.NORMAL.value}": 28,
    f"{card.COLOR.GREEN.value} | {card.VALUE.DRAW4.value}": 29,
    f"{card.COLOR.BLUE.value} | {card.VALUE.ZERO.value}": 30,
    f"{card.COLOR.BLUE.value} | {card.VALUE.ONE.value}": 31,
    f"{card.COLOR.BLUE.value} | {card.VALUE.TWO.value}": 32,
    f"{card.COLOR.BLUE.value} | {card.VALUE.THREE.value}": 33,
    f"{card.COLOR.BLUE.value} | {card.VALUE.FOUR.value}": 34,
    f"{card.COLOR.BLUE.value} | {card.VALUE.FIVE.value}": 35,
    f"{card.COLOR.BLUE.value} | {card.VALUE.SIX.value}": 36,
    f"{card.COLOR.BLUE.value} | {card.VALUE.SEVEN.value}": 37,
    f"{card.COLOR.BLUE.value} | {card.VALUE.EIGHT.value}": 38,
    f"{card.COLOR.BLUE.value} | {card.VALUE.NINE.value}": 39,
    f"{card.COLOR.BLUE.value} | {card.VALUE.SKIP.value}": 40,
    f"{card.COLOR.BLUE.value} | {card.VALUE.REVERSE.value}": 41,
    f"{card.COLOR.BLUE.value} | {card.VALUE.DRAW2.value}": 42,
    f"{card.COLOR.BLUE.value} | {card.VALUE.NORMAL.value}": 43,
    f"{card.COLOR.BLUE.value} | {card.VALUE.DRAW4.value}": 44,
    f"{card.COLOR.YELLOW.value} | {card.VALUE.ZERO.value}": 45,
    f"{card.COLOR.YELLOW.value} | {card.VALUE.ONE.value}": 46,
    f"{card.COLOR.YELLOW.value} | {card.VALUE.TWO.value}": 47,
    f"{card.COLOR.YELLOW.value} | {card.VALUE.THREE.value}": 48,
    f"{card.COLOR.YELLOW.value} | {card.VALUE.FOUR.value}": 49,
    f"{card.COLOR.YELLOW.value} | {card.VALUE.FIVE.value}": 50,
    f"{card.COLOR.YELLOW.value} | {card.VALUE.SIX.value}": 51,
    f"{card.COLOR.YELLOW.value} | {card.VALUE.SEVEN.value}": 52,
    f"{card.COLOR.YELLOW.value} | {card.VALUE.EIGHT.value}": 53,
    f"{card.COLOR.YELLOW.value} | {card.VALUE.NINE.value}": 54,
    f"{card.COLOR.YELLOW.value} | {card.VALUE.SKIP.value}": 55,
    f"{card.COLOR.YELLOW.value} | {card.VALUE.REVERSE.value}": 56,
    f"{card.COLOR.YELLOW.value} | {card.VALUE.DRAW2.value}": 57,
    f"{card.COLOR.YELLOW.value} | {card.VALUE.NORMAL.value}": 58,
    f"{card.COLOR.YELLOW.value} | {card.VALUE.DRAW4.value}": 59,
    "DRAW": 60,    
}

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
    if(color == card.COLOR.BLUE.value):
        colorText = f'{TextCode.BLUE.value}{text}{TextCode.RESET.value}'
    elif(color == card.COLOR.RED.value):
        colorText = f'{TextCode.RED.value}{text}{TextCode.RESET.value}'
    elif(color == card.COLOR.YELLOW.value):
        colorText = f'{TextCode.YELLOW.value}{text}{TextCode.RESET.value}'
    elif(color == card.COLOR.GREEN.value):
        colorText = f'{TextCode.GREEN.value}{text}{TextCode.RESET.value}'
    elif(color == card.COLOR.WILD.value):
        colorText = f'{TextCode.GRAY.value}{text}{TextCode.RESET.value}'
    return colorText

def hand_to_dict(hand):
    handDict = {}
    for card in hand:
        if card.__repr__() not in handDict:
            handDict[card.__repr__()] = 1
        else:
            handDict[card.__repr__()] += 1
    return handDict

## given a list of cards, returns the state representation of it which is a 4x15 matrix
def hand_to_state_rep(hand):
    state_matrix = np.zeros((4, 15), dtype=int)
    hand = hand_to_dict(hand) 
    for c, count in hand.items():
        cardInfo = c.split(' | ')
        val = VALUE_MAP[cardInfo[1]]
        # put wild in each color row
        if cardInfo[0] == card.COLOR.WILD.value:
            for col in COLOR_MAP.values():
                state_matrix[col][val] += count
        else:
            color = COLOR_MAP[cardInfo[0]]
            state_matrix[color][val] += count
    return state_matrix

def state_rep_to_action_numbers_list(state_matrix):
    actionNumberList = []
    for i in range(len(state_matrix)):
        for j in range(len(state_matrix[i])):
            if state_matrix[i][j] >= 1:
                actionNumber = (i * len(state_matrix[i])) + j
                actionNumberList.append(actionNumber)
    return actionNumberList


# def hand_to_state_rep(state_matrix, hand):
#     state_matrix = np.zeros((5, 15), dtype=int)
#     hand = hand_to_dict(hand) 
#     for card, count in hand.items():
#         cardInfo = card.split(' | ')
#         color = COLOR_MAP[cardInfo[0]]
#         value = VALUE_MAP[cardInfo[1]]
#         state_matrix[color][value] += count
#     return state_matrix

def target_to_state_rep(state_matrix, target):
    target_info = target.split(' | ')
    color = COLOR_MAP[target_info[0]]
    value = VALUE_MAP[target_info[1]]
    state_matrix[color][value] = 1
    return state_matrix

def state_to_card_rep(state_matrix):
    pass

def card_to_action_number(c):
    return ACTION_MAP[c.__repr__()]

def card_rep_to_action_number(repr):
    return ACTION_MAP[repr]

#0-60
def action_to_card_rep(action):
    ## action 60 is draw
    if action == 60:
        return None
    
    key = next((k for k, v in ACTION_MAP.items() if v == action), None)

    ## if card is a wild card
    ## make sure divide by zero doesn't happen
    if not action in (0, 1, 2, 7):
        if ((int(action / 15) + 1)* 15) % action in (1, 2):
            keyColor = key.split(' | ')[0]
            key = key.replace(key.split(' | ')[0], card.COLOR.WILD.value)
            return key, keyColor
    return key, None