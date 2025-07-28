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


def main():
    agentIds = ['UnoAgent_0', 'UnoAgent_1', 'UnoAgent_2', 'UnoAgent_3']
    players = {id: player.Player(id) for id in agentIds}

    doLoopTest = False

    if doLoopTest:
        env_config= {
            "players": players,     # Pass any required env args here
            "hasHuman": False
        }
        RLLib = RLLibEnv.UnoRLLibEnv(env_config)
        RLLib.reset()

        game_loop = TestLoop()

        game_loop.start(1, RLLib)

    else:
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
                policies={"UnoAgent_0": PolicySpec(), "UnoAgent_1": PolicySpec(), "UnoAgent_2": PolicySpec(), "UnoAgent_3": PolicySpec()},
                policy_mapping_fn=lambda agent_id, episode, **kw: agent_id,
                policies_to_train=agentIds,  # Specify which policies to train
            )
            .framework("torch")
            .env_runners(num_env_runners=1)
            .training(replay_buffer_config={
                "type": "MultiAgentPrioritizedReplayBuffer",
                "capacity": 60000,
            })
            .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=DQNActionMaskModel.ActionMaskDQNTorchRLModule,
            ),
    )
        )

        dqn_w_custom_env = config.build_algo()
        dqn_w_custom_env.train()


if __name__ == "__main__":
    main()
