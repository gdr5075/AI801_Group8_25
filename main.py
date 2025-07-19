from uno_package import player, game, env, utils,deck, loop
import gymnasium as gym
import numpy as np
from pettingzoo.utils import AgentSelector, wrappers
import torch
import copy
import os
import random
from collections import deque
from datetime import datetime

import numpy as np
import torch
import wandb
import yaml
from tqdm import tqdm
from pettingzoo.classic import connect_four_v3

from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import create_population
from pettingzoo.test import api_test

def main():
    agentIds = ['UnoAgent_0', 'UnoAgent_1', 'UnoAgent_2', 'UnoAgent_3']
    # me = player.HumanPlayer('Zach')
    frodo = player.Player('Frodo')
    players = [player.Player('Smaug'), frodo, player.Player('Sauron'), player.Player('Gollum')]
    
    unoEnv = env.raw_env(players, False)
    # testLoop = loop.TestLoop()
    # testLoop.start(2, unoEnv)
    # x = torch.rand(5,3)
    # print(x)
    # print(torch.cuda.is_available())

    api_test(unoEnv, num_cycles=1_000_000)


if __name__ == "__main__":
    main()
