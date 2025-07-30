from uno_package import player, game, env, utils,deck, loop, card, DQNActionMaskModel
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
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from uno_package.loop import TestLoop
from ray import tune


def main():
    agentIds = ['UnoAgent_0', 'UnoAgent_1', 'UnoAgent_2', 'UnoAgent_3']
    players = {id: player.Player(id) for id in agentIds}

    doLoopTest = False

    ## feel free to change
    reward_values = {
        'draw2_uno': .5,
        'draw4_uno': .5,
        'skip_uno': .2,
        'reverse_from_uno': .2,
        'turn': -0.1,
        'win': 10.0,
        'lose': -10.0,
    }

    if doLoopTest:
        env_config= {
            "players": players,     # Pass any required env args here
            "hasHuman": False,
            "reward_values": reward_values,
        }
        RLLib = RLLibEnv.UnoRLLibEnv(env_config)
        RLLib.reset()

        game_loop = TestLoop()

        game_loop.start(1, RLLib)

    else:
        tune.register_env("UnoRLLibEnv", lambda config: RLLibEnv.UnoRLLibEnv(config))
        config = (
            DQNConfig()
            .environment(
                ## not sure if this is correct either, but we can use tune.register_env to register the custom environment if we need to
                env = "UnoRLLibEnv", #This cant be right.
                env_config= {
                    "players": players,     # Pass any required env args here
                    "hasHuman": False,
                    "reward_values": reward_values,
                }
            )
            .multi_agent(
                policies={"UnoAgent_0": PolicySpec(), "UnoAgent_1": PolicySpec(), "UnoAgent_2": PolicySpec(), "UnoAgent_3": PolicySpec()},
                policy_mapping_fn=lambda agent_id, episode, **kw: agent_id,
                policies_to_train=agentIds,  # Specify which policies to train
            )
            .framework("torch")
            .env_runners(num_env_runners=1)
            # .training(
            #     train_batch_size=32,
            #     gamma=0.99,
            #     lr=1e-3,
            #     replay_buffer_config={
            #         "capacity": 60000,
            #     },
            #     dueling=True,        # Enable dueling DQN
            #     double_q=True        # Enable Double Q-learning
            # )
            .rl_module(
                rl_module_spec=RLModuleSpec(
                    module_class=DQNActionMaskModel.ActionMaskDQNTorchRLModule,
                ),
            )
            # .resources(
            #     num_gpus=1,          # Set to 1 or more if using GPUs
            #     num_cpus_per_worker=1
            # )
        )

        dqn_w_custom_env = config.build_algo()
        dqn_w_custom_env.train()


if __name__ == "__main__":
    main()
