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
            policies={
                ## not quite sure what the first argument, policy_class is
                agent_id: PolicySpec(None, RLLibEnv.UnoRLLibEnv.observation_space(agent_id), RLLibEnv.UnoRLLibEnv.action_space(agent_id), {}) for agent_id in agentIds
            },
            policy_mapping_fn=AgentSelector(lambda agent_id: agent_id),
            policies_to_train=list(agentIds),  # Specify which policies to train
        )
        .framework("torch")
        .rollouts(
            num_rollout_workers=1,  # Increase for more parallelism
        )
        .training(replay_buffer_config={
            "type": "PrioritizedEpisodeReplayBuffer",
            "capacity": 60000,
            "alpha": 0.5,
            "beta": 0.5,
        })
    )

    dqn_w_custom_env = config.build_algo()
    dqn_w_custom_env.train()


if __name__ == "__main__":
    main()
