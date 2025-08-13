from uno_package import player, game, env, utils,deck, loop, card, DQNActionMaskModel, RLLibEnv, RLLibEnvSingleAgent, training
import torch
import random
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
import argparse

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    print("Welcome to the Uno AI training script")
    print("To train a model, run the script with --doTrain")
    print("    The default is DQN without action masking")
    print("To evaluate the most recent model, run the script with --doEvaluate")
    print("To find the model with the highest win rate, run the script with --findModel")
    print("To evaluate a specific model, run the script with --evaluateModel <model_name>")
    print("To train a model with action masking, run the script with --doTrain --doActionMask")
    print("To train a model with PPO, run the script with --doTrain --doPPO")
    print("To train a model with PPO and action masking, run the script with --doTrain --doPPO --doActionMask")
    print("--------------------------------")


    parser.add_argument(
        "--doEvaluate", "-e",
        action="store_true",
        help="To evaluate the most recent model, overidden by --evaluateModel"
    )

    parser.add_argument(
        "--evaluateModel", "-v",
        action="store",
        help="The model to evaluate, overrides --doEvaluate"
    )

    parser.add_argument(
        "--numGames", "-n",
        action="store",
        help="The number of games to evaluate, only used for --evaluateModel"
    )

    parser.add_argument(
        "--doPPO", "-p",
        action="store_true",
        help="To use PPO instead of DQN"
    )

    parser.add_argument(
        "--doActionMask", "-m",
        action="store_true",
        help="To include action mask in the model"
    )

    parser.add_argument(
        "--findModel", "-f",
        action="store_true",
        help="To find the model with the highest win rate, overidden by --evaluateModel or --doEvaluate"
    )

    parser.add_argument(
        "--doTrain", "-t",
        action="store_true",
        help="To train a model"
    )

    args = parser.parse_args()

    #tf.debugging.experimental.enable_dump_debug_info("~/ray_results", tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
    agentIds = ['UnoAgent_0', 'UnoAgent_1', 'UnoAgent_2', 'UnoAgent_3']
    players = {id: player.Player(id) for id in agentIds}


    ## feel free to change
    reward_values = {
        'draw2_uno': 2,
        'draw4_uno': 2,
        'skip_uno': 2,
        'reverse_from_uno': 2,
        'color_select' : 0.5,
        'turn': 4, #Reward for playing a card
        'draw_card': 0.1, # Draw card is a valid action, but not a good one
        'win': 40.0,
        'lose': 0,
        'invalid_action': -0.1,
    }

    if args.doEvaluate or args.evaluateModel:
        if args.evaluateModel:
            if args.numGames:
                playground.demo_multiple_games(args.evaluateModel, int(args.numGames))
            else:
                playground.demo_multiple_games(args.evaluateModel, 1000)
        else:
            playground.demo_multiple_games_latest_checkpoint()

    elif args.findModel:
        playground.find_highest_win_rate_checkpoint()
    elif args.doTrain:
        print("Training a model")
        print(torch.cuda.is_available())
        if args.doPPO:
            #TODO: Implement PPO
            print("PPO is not implemented yet")
            pass
        else:
            #print("DQN is implemented")
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
                .framework("torch")
                .env_runners(num_env_runners=4)
                .training(
                    train_batch_size=2048,
                    minibatch_size=1024,
                    gamma=0.99,
                    lr=0.0001,
                    replay_buffer_config={
                        "type": "PrioritizedEpisodeReplayBuffer",
                        "capacity": 60000,
                    }
                )
                .learners(
                    num_learners=0,
                )
                .resources(
                    num_gpus=1,
                )
            )
            if args.doActionMask:
                config.rl_module(
                    rl_module_spec=RLModuleSpec(
                        module_class=DQNActionMaskModel.ActionMaskDQNTorchRLModule,
                    )
                )

            config.validate()

            dqn_w_custom_env = config.build_algo()
            working = True
            while working:
                try:
                    training.train_multiple_iterations(dqn_w_custom_env, 100)
                    working = False
                except Exception as e:
                    print(f"Error: {e}")


if __name__ == "__main__":
    main()
