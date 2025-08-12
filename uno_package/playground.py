import torch
import os
import random
import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
from uno_package import player, RLLibEnvSingleAgent, utils, PlaygroundEnv
from pathlib import Path

def load_checkpoint(checkpoint_number=5):
    dir_path = os.path.dirname(os.path.realpath(__file__))+"/../"
    checkpoint_path = f"file://{dir_path}/checkpoints/checkpoint_{checkpoint_number}"
    try:
        saved_algorithm = Algorithm.from_checkpoint(path=checkpoint_path)
        print(f"Successfully loaded checkpoint")
        return saved_algorithm
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_number}: {e}")
        return None

def load_checkpoint_from_path(checkpoint_path):
    try:
        saved_algorithm = Algorithm.from_checkpoint(path=checkpoint_path)
        print(f"Successfully loaded checkpoint from {checkpoint_path}")
        return saved_algorithm
    except Exception as e:
        print(f"Error loading checkpoint at {checkpoint_path}: {e}")
        return None

def load_latest_checkpoint():
    dir_path = os.path.dirname(os.path.realpath(__file__))+"/../checkpoints/"
    checkpoint_path = get_latest_created_folder(dir_path)
    try:
        saved_algorithm = Algorithm.from_checkpoint(path=checkpoint_path)
        print(f"Successfully loaded checkpoint from {checkpoint_path}")
        return saved_algorithm
    except Exception as e:
        print(f"Error loading checkpoint at {checkpoint_path}: {e}")
        return None
    
def get_latest_created_folder(directory):
    # Get all subdirectories in the given directory
    subdirs = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    
    # Find the newest folder based on creation time
    latest_folder = max(subdirs, key=os.path.getctime)
    return latest_folder

def play_single_game(algorithm,env,players, num_random_players=3, verbose=True, file=None):


    
    env.set_randomize(True)
    obs, _ = env.reset()

    game_stats = {
        'turns': 0,
        'winner': None,
        'final_hand_counts': {},
        'trained_agent_won': False,
        'normal': 0,
        'replaced': 0,
        'win_counts': {},
        'total_hand_counts': {}
    }
    for i in players:
        game_stats['win_counts'][i] = 0
        game_stats['total_hand_counts'][i] = 0
    replaced = 0
    normal = 0
    
    policy = algorithm.get_module()
    # Game loop
    while not env.winning_player:

        current_player = env.current_player
        if file:
            file.write(f"Turn {game_stats['turns']}: {current_player}'s turn\n")
            file.write(f"Top card: {env.get_top_play_card()}\n")
            file.write(f"Current player hand - {env.get_player(current_player).hand}\n")
        
        game_stats['turns'] += 1
        
        if verbose:
            print(f"\nTurn {game_stats['turns']}: {current_player}'s turn")
            print(f"Top card: {env.get_top_play_card()}")
            print(f"Current player hand - {env.get_player(current_player).hand}")
        
        # Get observation for current player
        obs = env.observe(current_player)
        
        # Choose action based on player type
        if current_player == 'TrainedAgent':
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            batch = {'obs': obs_tensor}
            output = policy._forward_inference(batch)
            #print(f"Output: {output}")
            action = output['actions'].item()  # Extract action from output
            if file:
                file.write(f"Action: {action}\n")
            if action not in env.get_valid_actions_for_player(players['TrainedAgent']):
                # print(f"Invalid action: {action}")
                # print(f"Valid actions: {env.get_valid_actions_for_player(players[current_player])}")
                if file:
                    file.write(f"Invalid action: {action}\n")
                    file.write(f"Valid actions: {env.get_valid_actions_for_player(players[current_player])}\n")
                action = players[current_player].get_action_sa(obs, env)
                replaced += 1
            else:
                normal += 1
        else:
            #TODO - This portion needs to be cleaned up / reworked to support other action types
            action = players[current_player].get_action_sa(obs, env)
        
        if verbose:
            card_rep = utils.action_to_card_rep(action)
            if(card_rep != None):
                action_card, chosen_color = card_rep
                print(f"{current_player} plays: {action_card}")
                if chosen_color:
                    print(f"Chosen color: {chosen_color}")
            else:
                print(f"{current_player} plays: Draw Card")

        

        # Take step
        obs, reward, done, truncated, info = env.step(action)
        
        if verbose:
            print(f"Reward: {reward}")
            print(f"Game direction: {'Clockwise' if env.isClockwise else 'Counter-clockwise'}")
        
        # Check for winner
        if env.winning_player:
            game_stats['winner'] = env.winning_player
            game_stats['trained_agent_won'] = (env.winning_player == 'TrainedAgent')
            game_stats['final_hand_counts'] = env.get_hand_counts()
            game_stats['win_counts'][env.winning_player] += 1
            game_stats['normal'] = normal
            game_stats['replaced'] = replaced
            normal = 0
            replaced = 0
            total_hand_counts = 0
            hands = env.get_hand_counts()
            for counts in hands:
                for player in counts:
                    if player != env.winning_player:
                        total_hand_counts += counts[player]
            game_stats['total_hand_counts'][env.winning_player] += (total_hand_counts)

            if verbose:
                print(f"\n🎉 Game Over! Winner: {env.winning_player}")
                print(f"Final hand counts: {game_stats['final_hand_counts']}")
                print("=" * 50)
    
    return game_stats

def play_multiple_games(algorithm, num_games=10, num_random_players=3):
    
    results = {
        'total_games': num_games,
        'trained_agent_wins': 0,
        'random_player_wins': 0,
        'average_turns': 0,
        'win_rate': 0.0,
        'game_results': [],
        'normal': 0,
        'replaced': 0,
        'win_counts': {},
        'total_hand_counts': {}
    }

    f = open("playground.txt", "w")

    total_turns = 0
    # Create players
    agentIds = ['TrainedAgent'] + [f'RandomPlayer_{i}' for i in range(num_random_players)]
    players = {id: player.Player(id) for id in agentIds}


    for i in agentIds:
        results['win_counts'][i] = 0
        results['total_hand_counts'][i] = 0

    reward_values = {
        'draw2_uno': 2,
        'draw4_uno': 2,
        'skip_uno': 2,
        'reverse_from_uno': 2,
        'turn': -0.1,
        'win': 10.0,
        'lose': -5,
    }
    
    env_config = {
        "players": players,
        "hasHuman": False,
        "reward_values": reward_values,
    }
    env = PlaygroundEnv.PlaygroundEnv(env_config)
    
    for game_num in range(num_games):
        
        #if (game_num % 100 == 0):
        print(f"Executing game number {game_num}")
        f.write(f"Executing game number {game_num}\n")  

        game_stats = play_single_game(algorithm, env, players, num_random_players, False, file=f)
        results['game_results'].append(game_stats)
        
        if game_stats['trained_agent_won']:
            results['trained_agent_wins'] += 1
        else:
            results['random_player_wins'] += 1
        
        total_turns += game_stats['turns']
        results['normal'] += game_stats['normal']
        results['replaced'] += game_stats['replaced']
        results['win_counts'][game_stats['winner']] += 1
        results['total_hand_counts'][game_stats['winner']] += game_stats['total_hand_counts'][game_stats['winner']]
        
    results['average_turns'] = total_turns / num_games
    results['trained_win_rate'] = results['trained_agent_wins'] / num_games
    
    # Print summary
    print("Game Stats")
    print(f"{'='*30}")
    print(f"Total games played: {num_games}")
    print(f"Trained agent wins: {results['trained_agent_wins']}")
    print(f"Random player wins: {results['random_player_wins']}")
    print(f"Win rate: {results['trained_win_rate']}")
    print(f"Average turns per game: {results['average_turns']:.1f}")

    for i in results['win_counts']:
        print(f"{i} wins: {results['win_counts'][i]}")

    for i in results['total_hand_counts']:
        print(f"{i} average opponenthand count: {results['total_hand_counts'][i]/(results['win_counts'][i]+0.001)}")
    print(f"Correct action ratio: {results['normal']/(results['normal']+results['replaced'])}")
    print(f"{'='*50}")
    
    return results


def demo_single_game(checkpoint_num):
    algorithm = load_latest_checkpoint()
    play_single_game(algorithm, num_random_players=3, verbose=False)

def demo_multiple_games(checkpoint_num):
    algorithm = load_checkpoint(checkpoint_num)
    play_multiple_games(algorithm, num_games = 1000, num_random_players=3)

def demo_multiple_games_latest_checkpoint():
    algorithm = load_latest_checkpoint()
    play_multiple_games(algorithm, num_games = 1000, num_random_players=3)

def find_highest_win_rate_checkpoint():
    highest_win_rate = 0
    dir_path = os.path.dirname(os.path.realpath(__file__))+"/../checkpoints/"
    for dir in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, dir)):
            checkpoint_path = os.path.join(dir_path, dir)
            algorithm = load_checkpoint_from_path(checkpoint_path)
            if algorithm is None:
                continue
            results = play_multiple_games(algorithm, num_games = 250, num_random_players=3)
            if results['trained_win_rate'] > highest_win_rate:
                highest_win_rate = results['trained_win_rate']
                highest_win_rate_checkpoint = dir
    print(f"Highest win rate checkpoint: {highest_win_rate_checkpoint} with win rate: {highest_win_rate}")



