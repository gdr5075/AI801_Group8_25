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

import numpy as np
import torch
from ray.rllib.algorithms.dqn import DQNConfig

def main():
    agentIds = ['UnoAgent_0', 'UnoAgent_1', 'UnoAgent_2', 'UnoAgent_3']
    # me = player.HumanPlayer('Zach')
    frodo = player.Player('Frodo')
    players = [player.Player('Smaug'), frodo, player.Player('Sauron'), player.Player('Gollum')]
    
    RLLibEnv.UnoAgentSelector

    # unoEnv = env.raw_env(players, False)
    # unoEnv.reset()

    # RLLib = RLLibEnv.UnoRLLibEnv(players, False)

    config = (
        DQNConfig()
        .environment(
            ## not sure if this is correct either, but we can use tune.register_env to register the custom environment if we need to
            env = RLLibEnv.UnoRLLibEnv, #This cant be right.
            env_config= {
                "players": agentIds,     # Pass any required env args here
                "hasHuman": False
            }
        )
        .multi_agent(
            policies={"UnoAgent_0", "UnoAgent_1", "UnoAgent_2", "UnoAgent_3"},
            policy_mapping_fn=lambda agent_id, episode, **kw: agent_id,
            policies_to_train=agentIds,  # Specify which policies to train
        )
        .framework("torch")
        .env_runners(num_env_runners=2)
        .training(replay_buffer_config={
            "type": "MultiAgentReplayBuffer",
            "capacity": 60000,
        })
    )

    dqn_w_custom_env = config.build_algo()
    dqn_w_custom_env.train()


if __name__ == "__main__":
    main()
