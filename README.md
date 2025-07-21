# AI801_Group8_25
Project Repo for AI801 Group8 for Summer25

'pip instal pipenv'
'pip install requests'
so we can manage dependencies better
use pipenv to install packages
'pipenv run python main.py' to run the program


Using pettingzoo which is a multi-agent implementation of gymnasium
We are AEC env, not parallel
gymnasium https://gymnasium.farama.org/ 
pettingzoo https://pettingzoo.farama.org/content/basic_usage/ 
agileRL is what pettingzoo uses for DQN https://pettingzoo.farama.org/tutorials/agilerl/DQN/ 
agileRL https://docs.agilerl.com/en/latest/api/algorithms/dqn.html 


Learning:
multi-agent experience replay buffer for off policy training
transition is a named tuple representing a single transition in our environment. It essentially maps (state, action) pairs to their (next_state, reward) result, with the state being the screen difference image as described later on.
