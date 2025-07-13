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


TODO:
Probably need to change how wilds work. This would avoid the need for the agent to decide a card and then decide a color.
Instead should add wild cards to each color as potential actions
Let's say player has 1 normal wild. In the state_matrix, add a 1 in each row for the normal wild column. This would add 8 more actions and remove the two regular wild actions currently in the action_map. This would remove 1 row from the state matrices, making it a 4, 15. Requires refactoring some methods that convert to/from state matrices in utils.