from tqdm import tqdm

class Loop():
    def __init__(self):
        pass

    def start(episodes, env):
        pass

class TrainingLoop(Loop):
    def __init__(self):
        pass

    def start(episodes, env):
        for episode in range(episodes):
            obs, info = env.reset()
            done = False
            while not done:
                
                ## below should be in step()
                nextPlayer = self.get_next_player()
                action = nextPlayer.get_action(self, )
                
                # if tuple wild was played and color chosen
                if isinstance(action, tuple):
                    pass


class TestLoop(Loop):
    def __init__(self):
        pass
    
    def start(episodes, env):
        for episode in range(episodes):
            while not env.winning_player == None:
                
                currentPlayer  = env.agent_selection
                observation = env.observe(currentPlayer)
                currentReward = env.rewards[currentPlayer]
                action = currentPlayer.get_action(observation)
                nextObservation = env.observe(currentPlayer)
                reward = env.rewards[currentPlayer] - currentReward
                env.step(action)
                currentPlayer.update(observation, action, )
                
                # if tuple wild was played and color chosen
                if isinstance(action, tuple):
                    pass