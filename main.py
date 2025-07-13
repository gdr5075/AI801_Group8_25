from uno_package import player, game, env, utils,deck
import gymnasium as gym
import numpy as np
from pettingzoo.utils import AgentSelector, wrappers

def main():
    me = player.HumanPlayer('Zach')
    players = [me, player.Player('Frodo'), player.Player('Sauron'), player.Player('Gollum')]
    #unoEnv = env.UnoEnvironment(players, True)
    #print(me.hand)
    #print(unoEnv.observe(me))
    agents = ['a', 'b', 'c', 'd']
    agent = env.UnoAgentSelector(agents)
    print(agent._current_agent)
    print(agent.selected_agent)
    test = agent.next(1)
    print(test)
    print(agent._current_agent)
    print(agent.selected_agent)
    test2 = agent.next(-1)
    print(test2)
    print(agent._current_agent)
    print(agent.selected_agent)
    test3 = agent.next(-1)
    print(test3)
    print(agent._current_agent)
    print(agent.selected_agent)
    test4 = agent.next(1)
    print(test4)
    print(agent._current_agent)
    print(agent.selected_agent)
    test5 = agent.next(1)
    print(test5)
    print(agent._current_agent)
    print(agent.selected_agent)

    # for i in range(0,20):
    #     me = player.HumanPlayer('Zach')
    #     players = [me, player.Player('Frodo'), player.Player('Sauron'), player.Player('Gollum')]
    #     game_instance = game.Game(players, True)
    #     game_instance.shuffle_players()
    #     game_instance.play()
        #plane = np.zeros((5, 15), dtype=int)
        #test = deck.UnoMainDeck()
        #cards = []
        #for _ in range(7):
        #    cards.append(test.pop())
        #print(cards)
        #print(utils.hand_to_plane(plane, cards))

if __name__ == "__main__":
    main()
