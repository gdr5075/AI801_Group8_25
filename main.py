from uno_package import player, game, env, utils,deck, loop, card
import gymnasium as gym
import numpy as np
from pettingzoo.utils import AgentSelector, wrappers
import torch
import copy
import os
import random
from collections import deque
from datetime import datetime

from uno_package import RLLibEnv
from uno_package.loop import TestLoop

from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.policy.policy import PolicySpec

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
from agilerl.components.data import Transition


def main():
    agentIds = ['UnoAgent_0', 'UnoAgent_1', 'UnoAgent_2', 'UnoAgent_3']
    # me = player.HumanPlayer('Zach')

    frodo = player.Player('Frodo')
    players = [player.Player('Smaug'), frodo, player.Player('Sauron'), player.Player('Gollum')]

    # unoEnv = env.raw_env(players, False)
    # unoEnv.reset()
    print(f'{players}')
#    test_players = 
    env_config= {
                 "players": players,     # Pass any required env args here
                 "hasHuman": False
             }

    RLLib = RLLibEnv.UnoRLLibEnv(env_config)
    
    loop  = TestLoop()

    loop.start(1, RLLib)

    # config = (
    #     DQNConfig()
    #     .environment(
    #         ## not sure if this is correct either, but we can use tune.register_env to register the custom environment if we need to
    #         env = RLLibEnv.UnoRLLibEnv, #This cant be right.
    #         env_config= {
    #             "players": players,     # Pass any required env args here
    #             "hasHuman": False
    #         }
    #     )
    #     .multi_agent(
    #         policies={"UnoAgent_0", "UnoAgent_1", "UnoAgent_2", "UnoAgent_3"},
    #         policy_mapping_fn=lambda agent_id, episode, **kw: agent_id,
    #         policies_to_train=agentIds,  # Specify which policies to train
    #     )
    #     .framework("torch")
    #     .env_runners(num_env_runners=1)
    #     .training(replay_buffer_config={
    #         "type": "MultiAgentReplayBuffer",
    #         "capacity": 60000,
    #     })
    # )

    # dqn_w_custom_env = config.build_algo()
    # dqn_w_custom_env.train()


if __name__ == "__main__":
    main()
