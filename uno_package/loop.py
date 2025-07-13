from tqdm import tqdm

class Loop():
    def __init__(self):
        pass

    def start(episodes, env):
        pass

class TestLoop(Loop):
    def __init__(self):
        pass
    
    def start(episodes, env):
        for episode in range(episodes):
            while not env.winning_player == None:
                
                currentPlayer  = env.agent_selection
                observation = env.observe(currentPlayer)
                previousReward = env.rewards[currentPlayer]
                action = currentPlayer.get_action(observation)
                env.step(action)
                nextObservation = env.observe(currentPlayer)
                rewardFromAction = env.rewards[currentPlayer] - previousReward
                #currentPlayer.update(observation, action, )
                # if game over, update all agents
                
                # if tuple wild was played and color chosen
                if isinstance(action, tuple):
                    pass
                
            env.reset()