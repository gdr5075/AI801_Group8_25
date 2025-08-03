from uno_package import player, game, env, utils,deck, loop, card, DQNActionMaskModel, RLLibEnv, RLLibEnvSingleAgent, training
import torch
import random
from ray.rllib.algorithms.dqn import DQNConfig
import torch
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from uno_package.loop import TestLoop
from ray import tune
import tensorflow as tf
from ray.tune.schedulers import PopulationBasedTraining
import pprint
from ray.rllib.algorithms.algorithm import Algorithm
import json
import os
import uno_package.playground as playground
from datetime import datetime

def main():

    #tf.debugging.experimental.enable_dump_debug_info("~/ray_results", tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
    agentIds = ['UnoAgent_0', 'UnoAgent_1', 'UnoAgent_2', 'UnoAgent_3']
    players = {id: player.Player(id) for id in agentIds}

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
        'color_select' : 1,
        'turn': -0.1,
        'win': 10.0,
        'lose': -5,
    }

    if doLoopTest:
        env_config= {
            "players": players,     # Pass any required env args here
            "hasHuman": False,
            "reward_values": reward_values,
        }
        RLLib = RLLibEnvSingleAgent.UnoRLLibEnv(env_config)
        RLLib.reset()

        game_loop = TestLoop()

        game_loop.start(1, RLLib)

    elif doEvaluate:

        playground.demo_multiple_games_latest_checkpoint()
    
    elif continueTraining:
        dir_path = os.path.dirname(os.path.realpath(__file__))+"/checkpoints/"
        checkpoint_path = playground.get_latest_created_folder(dir_path)
        new_dqn = Algorithm.from_checkpoint(checkpoint_path)
        training.train_multiple_iterations(new_dqn, 5)

    else:
        if not doTune:
            print(torch.cuda.is_available())
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
                .env_runners(num_env_runners=4)
                .training(
                    # train_batch_size=2048,
                    # minibatch_size=1024,
                    gamma=0.95,
                    replay_buffer_config={
                        "capacity": 60000,
                    }
                )
                .rl_module(
                    rl_module_spec=RLModuleSpec(
                        module_class=DQNActionMaskModel.ActionMaskDQNTorchRLModule,
                    ),
                ).resources(
                    num_gpus=1,
                )
            )
            dqn_w_custom_env = config.build_algo()
            training.train_multiple_iterations(dqn_w_custom_env, 5)
        else:
            print(torch.cuda.is_available())
            
            stopping_criteria = {"training_iteration": 1, "episode_reward_mean": 2}
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
                .env_runners(num_env_runners=4)
                .training(
                    # train_batch_size=2048,
                    # minibatch_size=1024,
                    gamma=0.95,
                    replay_buffer_config={
                        "capacity": 60000,
                    }
                )
                .rl_module(
                    rl_module_spec=RLModuleSpec(
                        module_class=DQNActionMaskModel.ActionMaskDQNTorchRLModule,
                    ),
                ).resources(
                    num_gpus=1,
                )
            )

            tune.register_env("UnoRLLibEnv", lambda config: RLLibEnvSingleAgent.UnoRLLibEnv(config))

            tuner = tune.Tuner(
                "DQN",
                param_space=config,
                # run_config=tune.RunConfig(stop={"num_env_steps_sampled_lifetime": 4000}),
                run_config=tune.RunConfig(stop={"training_iteration": 1, "mean_accuracy": 0.8}),
            )
            results = tuner.fit()

            best_result = results.get_best_result()

            print("\nBest performing trial's final reported metrics:\n")

            metrics_to_print = [
                "episode_reward_mean",
                "episode_reward_max",
                "episode_reward_min",
                "episode_len_mean",
            ]
            pprint.pprint({k: v for k, v in best_result.metrics.items() if k in metrics_to_print})

            loaded_ppo = Algorithm.from_checkpoint(best_result.checkpoint)
            # See your trained policy in action
            # loaded_policy.compute_single_action(...)


if __name__ == "__main__":
    main()
