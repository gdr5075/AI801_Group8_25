from tqdm import tqdm

class Loop():
    def __init__(self):
        pass

    def start(self, episodes, env):
        pass

class TestLoop(Loop):
    def __init__(self):
        pass
    
    def start(self, episodes, env):
        for _ in range(episodes):
            env.reset()
            while env.winning_player == None:
                print(f'Top card is {env.get_top_play_card()}')
                direction = 1 if env.isClockwise else -1
                player_name  = env._agent_selector.next(direction)
                currentPlayer = env.players[player_name]
                env.current_player = player_name
                print(f'current player {player_name}')
                observation = env.observe(player_name)
                action = currentPlayer.get_action(observation)
                print('get_action')
                env.step(action)
                nextObservation = env.observe(player_name)
                reward = env.rewards[player_name]
                #currentPlayer.update(observation, action, )
            print(f'Game over: {env.players[env.winning_player].name} wins!')
