from uno_package import player, game, env, utils,deck
import gymnasium as gym
import numpy as np

def main():
    for i in range(0,20):
        me = player.HumanPlayer('Zach')
        players = [me, player.Player('Frodo'), player.Player('Sauron'), player.Player('Gollum')]
        game_instance = game.Game(players, True)
        game_instance.shuffle_players()
        game_instance.play()
        #plane = np.zeros((5, 15), dtype=int)
        #test = deck.UnoMainDeck()
        #cards = []
        #for _ in range(7):
        #    cards.append(test.pop())
        #print(cards)
        #print(utils.hand_to_plane(plane, cards))

if __name__ == "__main__":
    main()
