from uno_package import utils


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
                print(f'loop player from agent selector: {player_name}')
                currentPlayer = env.players[player_name]
                print(f'Current player: {currentPlayer.name}')
                env.current_player = player_name
                print(f'current player {player_name}')
                observation = env.observe(player_name)
                available_moves = utils.hand_to_state_rep(env.get_valid_moves_for_player(currentPlayer))
                action = currentPlayer.get_action(available_moves)
                print('get_action')
                action_dict = {currentPlayer.name : action}
                env.step(action_dict)
                #nextObservation = env.observe(player_name)
                #reward = env.rewards[player_name]
                #currentPlayer.update(observation, action, )
            print(f'Game over: {env.players[env.winning_player].name} wins!')
