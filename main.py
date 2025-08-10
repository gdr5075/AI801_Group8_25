from uno_package import player, DQNActionMaskModel, RLLibEnv, RLLibEnvSingleAgent, training, RLLibEnvNoDraws, Modules
import torch
import random
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.examples.rl_modules.action_masking_rl_module import (
    ActionMaskingTorchRLModule,
)
from ray.rllib.algorithms.ppo.torch import default_ppo_torch_rl_module
import torch
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from uno_package.loop import TestLoop
from ray import tune
import tensorflow as tf
from ray.tune.schedulers import PopulationBasedTraining
import pprint
from ray.rllib.algorithms.algorithm import Algorithm
import os
import uno_package.playground as playground
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
import gymnasium as gym
import numpy as np
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec

def main():
    agentIds = ['UnoAgent_0', 'UnoAgent_1', 'UnoAgent_2', 'UnoAgent_3']
    #players = {id: player.Player(id) for id in agentIds}
    players = {
        'UnoAgent_0': player.Player('UnoAgent_0'),
        'UnoAgent_1': player.Player('UnoAgent_1'),
        'UnoAgent_2': player.Player('UnoAgent_2'),
        'UnoAgent_3': player.Player('UnoAgent_3'),
    }

    doEvaluate = False
    doLoopTest = False
    doTune = False
    continueTraining = True

    ## feel free to change
    reward_values = {
        'draw2_uno': 2,
        'draw4_uno': 2,
        'skip_uno': 2,
        'reverse_from_uno': 2,
        'draw2_less_cards': .25,
        'draw4_less_cards': .25,
        'skip_less_cards': .25,
        'reverse_less_cards': .25,
        'draw2_more_cards': -.25,
        'draw4_more_cards': -.25,
        'skip_more_cards': -.25,
        'reverse_more_cards': -.25,
        'color_select' : .2,
        'turn': 0.5,
        'draw': -.05,
        'win': 20,
        'lose': -5,
        'invalid_action': -.1,
        'less_than_start': .1,
        'uno': 2,
        'bad_special_play': -1
    }

    env_config =  {
        "players": players,
        "hasHuman": False,
        "reward_values": reward_values,
    }

    if doLoopTest:
        RLLib = RLLibEnv.UnoRLLibEnv(env_config)
        RLLib.reset()

        game_loop = TestLoop()

        game_loop.start(1, RLLib)

    elif doEvaluate:

        #playground.demo_multiple_games_latest_checkpoint()
        playground.demo_multiple_games("1754845004.20482", False)
    
    elif continueTraining:
        config = (
            PPOConfig()
            .environment(
                ## not sure if this is correct either, but we can use tune.register_env to register the custom environment if we need to
                env = RLLibEnvSingleAgent.UnoRLLibEnv, #This cant be right.
                env_config= env_config
            )
            # .multi_agent(
            #     policies={"UnoAgent_0", "UnoAgent_1", "UnoAgent_2", "UnoAgent_3"},
            #             #{"UnoAgent_0", "UnoAgent_1"},
            #     policy_mapping_fn=lambda agent_id, episode, **kw: agent_id,
            #     policies_to_train=["UnoAgent_0"],  # Specify which policies to train
            # )
            .framework("torch")
            .env_runners(
                num_env_runners=0,
            )
            .training(
                # burn_in_len=2
                # train_batch_size=2048,
                # minibatch_size=1024,
                # num_atoms=51,
                gamma=0.5,
                lambda_=.95,
                lr=.001,
                train_batch_size=4000,
                entropy_coeff=0.005,
                # minibatch_size=256,
                # num_epochs=10,
                # train_batch_size_per_learner=5000,
                # replay_buffer_config={
                #     "type": "PrioritizedEpisodeReplayBuffer",
                #     "capacity": 60000,
                # },
                # double_q=True,
                # dueling=True,
                # n_step=3,
                # target_network_update_freq=1000,
                # grad_clip=40.0,
                # optimizer={
                #     "adam_epsilon": 1e-8
                # },
            ).rl_module(
                # rl_module_spec= MultiRLModuleSpec( rl_module_specs={
                #     "UnoAgent_0": RLModuleSpec(
                #         module_class=default_ppo_torch_rl_module,
                #     ),
                #     "UnoAgent_1": RLModuleSpec(
                #         module_class=default_ppo_torch_rl_module,
                #     ),
                #     "UnoAgent_2": RLModuleSpec(
                #         module_class=default_ppo_torch_rl_module,
                #     ),
                #     "UnoAgent_3": RLModuleSpec(
                #         module_class=default_ppo_torch_rl_module,
                #     )
                # },
                rl_module_spec=RLModuleSpec(
                    module_class=ActionMaskingTorchRLModule,
                ),
                model_config = {
                            "use_lstm": True,
                            "max_seq_len": 50,
                }
                #),
            ).resources(
                num_gpus=1,
            )
        )
        dir_path = os.path.dirname(os.path.realpath(__file__))+"/checkpoints/"
        checkpoint_path = playground.get_latest_created_folder(dir_path)
        #new_dqn: Algorithm = Algorithm.from_checkpoint(checkpoint_path)
        new_dqn: Algorithm = config.build_algo()
        new_dqn.restore(checkpoint_path)
        training.train_multiple_iterations(new_dqn, 1000)

    else:
        if not doTune:
            print(torch.cuda.is_available())
            config = (
                PPOConfig()
                .environment(
                    ## not sure if this is correct either, but we can use tune.register_env to register the custom environment if we need to
                    env = RLLibEnvSingleAgent.UnoRLLibEnv, #This cant be right.
                    env_config= env_config
                )
                # .multi_agent(
                #     policies={"UnoAgent_0", "UnoAgent_1", "UnoAgent_2", "UnoAgent_3"},
                #             #{"UnoAgent_0", "UnoAgent_1"},
                #     policy_mapping_fn=lambda agent_id, episode, **kw: agent_id,
                #     policies_to_train=["UnoAgent_0"],  # Specify which policies to train
                # )
                .framework("torch")
                .env_runners(
                    num_env_runners=0,
                )
                .training(
                    # burn_in_len=2
                    # train_batch_size=2048,
                    # minibatch_size=1024,
                    # num_atoms=51,
                    gamma=0.99,
                    lambda_=.95,
                    lr=5e-5,
                    train_batch_size=4000,
                    entropy_coeff=0.01,
                    # minibatch_size=256,
                    # num_epochs=10,
                    # train_batch_size_per_learner=5000,
                    # replay_buffer_config={
                    #     "type": "PrioritizedEpisodeReplayBuffer",
                    #     "capacity": 60000,
                    # },
                    # double_q=True,
                    # dueling=True,
                    # n_step=3,
                    # target_network_update_freq=1000,
                    # grad_clip=40.0,
                    # optimizer={
                    #     "adam_epsilon": 1e-8
                    # },
                ).rl_module(
                    # rl_module_spec= MultiRLModuleSpec( rl_module_specs={
                    #     "UnoAgent_0": RLModuleSpec(
                    #         module_class=default_ppo_torch_rl_module,
                    #     ),
                    #     "UnoAgent_1": RLModuleSpec(
                    #         module_class=default_ppo_torch_rl_module,
                    #     ),
                    #     "UnoAgent_2": RLModuleSpec(
                    #         module_class=default_ppo_torch_rl_module,
                    #     ),
                    #     "UnoAgent_3": RLModuleSpec(
                    #         module_class=default_ppo_torch_rl_module,
                    #     )
                    # },
                    rl_module_spec=RLModuleSpec(
                        module_class=action_masking_rl_module,
                    ),
                    model_config = {
                                "use_lstm": True,
                                "max_seq_len": 50,
                    }
                    #),
                ).resources(
                    num_gpus=1,
                )
            )
            dqn_w_custom_env = config.build_algo()
            training.train_multiple_iterations(dqn_w_custom_env, 10000)
        else:
            print(torch.cuda.is_available())
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
            stopping_criteria = {"training_iteration": 5}
            config = (
                DQNConfig()
                .environment(
                    ## not sure if this is correct either, but we can use tune.register_env to register the custom environment if we need to
                    env = RLLibEnvSingleAgent.UnoRLLibEnv, #This cant be right.
                    env_config= {
                        "players": players,     # Pass any required env args here
                        "hasHuman": False,
                        "reward_values": reward_values,
                    }
                )
                # .multi_agent(
                #     policies={"UnoAgent_0", "UnoAgent_1", "UnoAgent_2", "UnoAgent_3"},
                #     policy_mapping_fn=lambda agent_id, episode, **kw: agent_id,
                #     policies_to_train=agentIds,  # Specify which policies to train
                # )
                .framework("torch")
                .env_runners(num_env_runners=4, num_cpus_per_env_runner=2)
                .training(
                    # These params are tuned from a fixed starting value.
                    gamma=.95,
                    lr=1e-4,
                    grad_clip=1,
                    # These params start off randomly drawn from a set.
                    num_epochs=tune.choice([10, 20, 30]),
                    minibatch_size=tune.choice([128, 512, 2048]),
                    train_batch_size_per_learner=tune.choice([10000, 20000, 40000]),
                )
                .rl_module(
                    rl_module_spec=RLModuleSpec(
                        module_class=DQNActionMaskModel.ActionMaskDQNTorchRLModule,
                    ),
                    model_config=DefaultModelConfig(free_log_std=True),
                ).resources(
                    num_gpus=1,
                )
            )

            tune.register_env("UnoRLLibEnv", lambda config: RLLibEnvSingleAgent.UnoRLLibEnv(config))

            tuner = tune.Tuner(
                'DQN',
                param_space=config,
                # run_config=tune.RunConfig(stop={"num_env_steps_sampled_lifetime": 4000}),
                run_config=tune.RunConfig(
                    stop=stopping_criteria,
                    checkpoint_config=tune.CheckpointConfig(
                    checkpoint_at_end=True,
                    checkpoint_frequency=10,
                ),),
            )
            results = tuner.fit()

            best_result = results.get_best_result()
            print(results.get_best_result().metrics)
            print(best_result.best_checkpoints)
            print(best_result.config)

            print("\nBest performing trial's final reported metrics:\n")

            metrics_to_print = [
                "episode_return_mean",
                "episode_return_max",
                "episode_return_min",
                "episode_len_mean",
            ]
            pprint.pprint({k: v for k, v in best_result.metrics.items() if k in metrics_to_print})

            loaded_ppo = Algorithm.from_checkpoint(best_result.checkpoint)
            # See your trained policy in action
            # loaded_policy.compute_single_action(...)


if __name__ == "__main__":
    main()
