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
                currentPlayer  = env.agent_selection
                print(f'current player {currentPlayer.name}')
                observation = env.observe(currentPlayer)
                action = currentPlayer.get_action(observation)
                print('get_action')
                env.step(action)
                nextObservation = env.observe(currentPlayer)
                reward = env.rewards[currentPlayer]
                #currentPlayer.update(observation, action, )
                # if game over, update all agents
                
                # if tuple wild was played and color chosen
                if isinstance(action, tuple):
                    pass