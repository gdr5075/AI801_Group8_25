from uno_package import player, game, env, utils,deck, loop
import gymnasium as gym
import numpy as np
from pettingzoo.utils import AgentSelector, wrappers

def main():
    me = player.HumanPlayer('Zach')
    frodo = player.Player('Frodo')
    players = [player.Player('Smaug'), frodo, player.Player('Sauron'), player.Player('Gollum')]
    
    unoEnv = env.raw_env(players, False)
    testLoop = loop.TestLoop()
    testLoop.start(2, unoEnv)

if __name__ == "__main__":
    main()
