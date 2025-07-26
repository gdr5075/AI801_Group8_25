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

from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.policy.policy import PolicySpec

import numpy as np
import torch
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.connectors.env_to_module import FlattenObservations


def main():
    agentIds = ['UnoAgent_0', 'UnoAgent_1', 'UnoAgent_2', 'UnoAgent_3']
    players = [player.Player(id) for id in agentIds]
    # unoEnv = env.raw_env(players, False)
    # unoEnv.reset()

    # RLLib = RLLibEnv.UnoRLLibEnv(players, False)

    config = (
        DQNConfig()
        .environment(
            ## not sure if this is correct either, but we can use tune.register_env to register the custom environment if we need to
            env = RLLibEnv.UnoRLLibEnv, #This cant be right.
            env_config= {
                "players": players,     # Pass any required env args here
                "hasHuman": False
            }
        )
        .multi_agent(
            policies={"UnoAgent_0", "UnoAgent_1", "UnoAgent_2", "UnoAgent_3"},
            policy_mapping_fn=lambda agent_id, episode, **kw: agent_id,
            policies_to_train=agentIds,  # Specify which policies to train
        )
        .framework("torch")
        .env_runners(num_env_runners=1)
        .training(replay_buffer_config={
            "type": "MultiAgentReplayBuffer",
            "capacity": 60000,
        })
    )

    dqn_w_custom_env = config.build_algo()
    dqn_w_custom_env.train()


if __name__ == "__main__":
    main()
