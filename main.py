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
import tensorflow as tf
from ray.tune.schedulers import PopulationBasedTraining
import pprint
from ray.rllib.algorithms.algorithm import Algorithm

def main():

    tf.debugging.experimental.enable_dump_debug_info("~/ray_results", tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
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
        # Postprocess the perturbed config to ensure it's still valid
        def explore(config):
            # ensure we collect enough timesteps to do sgd
            if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
                config["train_batch_size"] = config["sgd_minibatch_size"] * 2
            # ensure we run at least one sgd iter
            if config["num_sgd_iter"] < 1:
                config["num_sgd_iter"] = 1
            return config

        hyperparam_mutations = {
        "clip_param": lambda: random.uniform(0.01, 0.5),
        "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
        "num_epochs": lambda: random.randint(1, 30),
        "minibatch_size": lambda: random.randint(128, 16384),
        "train_batch_size_per_learner": lambda: random.randint(2000, 160000),
    }

        pbt = PopulationBasedTraining(
            time_attr="time_total_s",
            perturbation_interval=120,
            resample_probability=0.25,
            # Specifies the mutations of these hyperparams
            hyperparam_mutations=hyperparam_mutations,
            custom_explore_fn=explore,
        )

        stopping_criteria = {"training_iteration": 100, "episode_reward_mean": 300}
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
            .training(
                # These params are tuned from a fixed starting value.
                gamma=0.99,
                lr=1e-4,
                # These params start off randomly drawn from a set.
                num_epochs=tune.choice([10, 20, 30]),
                minibatch_size=tune.choice([128, 512, 2048]),
                train_batch_size_per_learner=tune.choice([10000, 20000, 40000]),
            )
            .rl_module(
                rl_module_spec=RLModuleSpec(
                    module_class=DQNActionMaskModel.ActionMaskDQNTorchRLModule,
                ),
            )
            .resources(
                num_gpus=1,          # Set to 1 or more if using GPUs
            )
        )

        tuner = tune.Tuner(
        "DQN",
        tune_config=tune.TuneConfig(
            metric="env_runners/episode_return_mean",
            mode="max",
            scheduler=pbt,
            num_samples=1
        ),
        param_space=config,
        run_config=tune.RunConfig(stop=stopping_criteria),
    )
    results = tuner.fit()

    best_result = results.get_best_result()

    print("Best performing trial's final set of hyperparameters:\n")
    pprint.pprint(
        {k: v for k, v in best_result.config.items() if k in hyperparam_mutations}
    )

    print("\nBest performing trial's final reported metrics:\n")

    metrics_to_print = [
        "episode_reward_mean",
        "episode_reward_max",
        "episode_reward_min",
        "episode_len_mean",
    ]
    pprint.pprint({k: v for k, v in best_result.metrics.items() if k in metrics_to_print})

    loaded_ppo = Algorithm.from_checkpoint(best_result.checkpoint)
    loaded_policy = loaded_ppo.get_policy()

    # See your trained policy in action
    # loaded_policy.compute_single_action(...)


if __name__ == "__main__":
    main()
