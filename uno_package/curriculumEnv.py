import random
from typing import List, Tuple, Optional

from pettingzoo import AECEnv
from uno_package import utils

from agilerl.components.data import Transition
from agilerl.components.replay_buffer import ReplayBuffer
from tqdm import tqdm
import numpy as np

class CurriculumEnv:
   """Wrapper around environment to modify reward for curriculum learning.

   :param env: Environment to learn in
   :type env: PettingZoo-style environment
   :param lesson: Lesson settings for curriculum learning
   :type lesson: dict
   """

   def __init__(self, env: AECEnv, lesson: dict):
      self.env = env
      self.lesson = lesson

   def fill_replay_buffer(
      self, memory: ReplayBuffer, opponent: "Opponent"
   ) -> ReplayBuffer:
      """Fill the replay buffer with experiences collected by taking random actions in the environment.

      :param memory: Experience replay buffer
      :type memory: AgileRL experience replay buffer
      :param opponent: Opponent to train against
      :type opponent: Opponent
      :return: Filled replay buffer
      :rtype: ReplayBuffer
      """
      print("Filling replay buffer ...")

      pbar = tqdm(total=memory.max_size)
      while len(memory) < memory.max_size:
            mem_full = len(memory)
            self.reset()  # Reset environment at start of episode
            observation, reward, done, truncation, _ = self.last()

            (
               p1_state,
               p1_state_flipped,
               p1_action,
               p1_next_state,
               p1_next_state_flipped,
            ) = (None, None, None, None, None)
            done, truncation = False, False

            while not self.env.winning_player:
               
               print(f'Top card is {self.env.get_top_play_card()}')
               currentPlayer  = self.env.agent_selection
               print(f'current player {currentPlayer.name}')
               observation = self.env.observe(currentPlayer)
               actionNum = self.env.action_space(currentPlayer).sample(utils.available_moves_to_action_mask(observation['observation']['available_moves']))
               self.env.step(actionNum)
               nextObservation = self.env.observe(currentPlayer)
               reward = self.env.rewards[currentPlayer]

               ## information for transition
               ## just making one big 2d array since transition want np arrays
               obsToAdd = np.zeros(15)
               obsToAdd[0] = utils.card_to_action_number(observation['observation']['top_card'])
               obsToAdd[1] = utils.color_to_number(observation['observation']['chosen_color'])
               obsToAdd[2] = observation['observation']['direction']
               obsToAdd[3] = observation['observation']['direction'][0]
               obsToAdd[4] = observation['observation']['direction'][1]
               obsToAdd[5] = observation['observation']['direction'][2]
               obsToAdd[6] = observation['observation']['direction'][3]
               nextObsToAdd = np.zeros(15)
               nextObsToAdd[0] = utils.card_to_action_number(observation['observation']['top_card'])
               nextObsToAdd[1] = utils.color_to_number(observation['observation']['chosen_color'])
               nextObsToAdd[2] = observation['observation']['direction']
               nextObsToAdd[3] = observation['observation']['direction'][0]
               nextObsToAdd[4] = observation['observation']['direction'][1]
               nextObsToAdd[5] = observation['observation']['direction'][2]
               nextObsToAdd[6] = observation['observation']['direction'][3]
               transition = Transition(
                  obs=(np.vstack([observation['observation'][currentPlayer], obsToAdd])),
                  action=[actionNum],
                  next_obs=(np.vstack([nextObservation['observation'][currentPlayer], nextObsToAdd])),
                  reward=[reward],
                  done=[False])
               memory.add()
                #currentPlayer.update(observation, action, )
               # Player 0's turn
               if opponent_first:
                  p0_action = self.env.action_space(currentPlayer).sample(utils.available_moves_to_action_mask(observation['observation']['available_moves']))
               else:
                  if self.lesson["warm_up_opponent"] == "random":
                        p0_action = opponent.get_action(
                           p0_action_mask, p1_action, self.lesson["block_vert_coef"]
                        )
                  else:
                        p0_action = opponent.get_action(player=0)
               self.step(p0_action)  # Act in environment
               observation, env_reward, done, truncation, _ = self.last()
               p0_next_state, p0_next_state_flipped = transform_and_flip(
                  observation, player=0
               )

               if done or truncation:
                  reward = self.reward(done=True, player=0)
                  transition = Transition(
                        obs=np.concatenate(
                           (p0_state, p1_state, p0_state_flipped, p1_state_flipped)
                        ),
                        action=np.array(
                           [p0_action, p1_action, 6 - p0_action, 6 - p1_action]
                        ),
                        reward=np.array(
                           [
                              reward,
                              LESSON["rewards"]["lose"],
                              reward,
                              LESSON["rewards"]["lose"],
                           ]
                        ),
                        next_obs=np.concatenate(
                           (
                              p0_next_state,
                              p1_next_state,
                              p0_next_state_flipped,
                              p1_next_state_flipped,
                           )
                        ),
                        done=np.array([done, done, done, done]),
                        batch_size=[4],
                  )
                  memory.add(transition.to_tensordict(), is_vectorised=True)
               else:  # Play continues
                  if p1_state is not None:
                        reward = self.reward(done=False, player=1)
                        transition = Transition(
                           obs=np.concatenate((p1_state, p1_state_flipped)),
                           action=np.array([p1_action, 6 - p1_action]),
                           reward=np.array([reward, reward]),
                           next_obs=np.concatenate(
                              (p1_next_state, p1_next_state_flipped)
                           ),
                           done=np.array([done, done]),
                           batch_size=[2],
                        )
                        memory.add(transition.to_tensordict(), is_vectorised=True)

                  # Player 1's turn
                  p1_action_mask = observation["action_mask"]
                  p1_state, p1_state_flipped = transform_and_flip(
                        observation, player=1
                  )
                  if not opponent_first:
                        p1_action = self.env.action_space("player_1").sample(
                           p1_action_mask
                        )
                  else:
                        if self.lesson["warm_up_opponent"] == "random":
                           p1_action = opponent.get_action(
                              p1_action_mask, p0_action, LESSON["block_vert_coef"]
                           )
                        else:
                           p1_action = opponent.get_action(player=1)
                  self.step(p1_action)  # Act in environment
                  observation, env_reward, done, truncation, _ = self.last()
                  p1_next_state, p1_next_state_flipped = transform_and_flip(
                        observation, player=1
                  )

                  if done or truncation:
                        reward = self.reward(done=True, player=1)
                        transition = Transition(
                           obs=np.concatenate(
                              (p0_state, p1_state, p0_state_flipped, p1_state_flipped)
                           ),
                           action=np.array(
                              [p0_action, p1_action, 6 - p0_action, 6 - p1_action]
                           ),
                           reward=np.array(
                              [
                                    LESSON["rewards"]["lose"],
                                    reward,
                                    LESSON["rewards"]["lose"],
                                    reward,
                              ]
                           ),
                           next_obs=np.concatenate(
                              (
                                    p0_next_state,
                                    p1_next_state,
                                    p0_next_state_flipped,
                                    p1_next_state_flipped,
                              )
                           ),
                           done=np.array([done, done, done, done]),
                           batch_size=[4],
                        )
                        memory.add(transition.to_tensordict(), is_vectorised=True)
                  else:  # Play continues
                        reward = self.reward(done=False, player=0)
                        transition = Transition(
                           obs=np.concatenate((p0_state, p0_state_flipped)),
                           action=np.array([p0_action, 6 - p0_action]),
                           reward=np.array([reward, reward]),
                           next_obs=np.concatenate(
                              (p0_next_state, p0_next_state_flipped)
                           ),
                           done=np.array([done, done]),
                           batch_size=[2],
                        )
                        memory.add(transition.to_tensordict(), is_vectorised=True)

            pbar.update(len(memory) - mem_full)
      pbar.close()
      print("Replay buffer warmed up.")
      return memory

   def check_winnable(self, lst: List[int], piece: int) -> bool:
      """Checks if four pieces in a row represent a winnable opportunity, e.g. [1, 1, 1, 0] or [2, 0, 2, 2].

      :param lst: List of pieces in row
      :type lst: List
      :param piece: Player piece we are checking (1 or 2)
      :type piece: int
      """
      return lst.count(piece) == 3 and lst.count(0) == 1

   def check_vertical_win(self, player: int) -> bool:
      """Checks if a win is vertical.

      :param player: Player who we are checking, 0 or 1
      :type player: int
      """
      board = np.array(self.env.env.board).reshape(6, 7)
      piece = player + 1

      column_count = 7
      row_count = 6

      # Check vertical locations for win
      for c in range(column_count):
            for r in range(row_count - 3):
               if (
                  board[r][c] == piece
                  and board[r + 1][c] == piece
                  and board[r + 2][c] == piece
                  and board[r + 3][c] == piece
               ):
                  return True
      return False

   def check_three_in_row(self, player: int) -> int:
      """Checks if there are three pieces in a row and a blank space next, or two pieces - blank - piece.

      :param player: Player who we are checking, 0 or 1
      :type player: int
      """
      board = np.array(self.env.env.board).reshape(6, 7)
      piece = player + 1

      # Check horizontal locations
      column_count = 7
      row_count = 6
      three_in_row_count = 0

      # Check vertical locations
      for c in range(column_count):
            for r in range(row_count - 3):
               if self.check_winnable(board[r : r + 4, c].tolist(), piece):
                  three_in_row_count += 1

      # Check horizontal locations
      for r in range(row_count):
            for c in range(column_count - 3):
               if self.check_winnable(board[r, c : c + 4].tolist(), piece):
                  three_in_row_count += 1

      # Check positively sloped diagonals
      for c in range(column_count - 3):
            for r in range(row_count - 3):
               if self.check_winnable(
                  [
                        board[r, c],
                        board[r + 1, c + 1],
                        board[r + 2, c + 2],
                        board[r + 3, c + 3],
                  ],
                  piece,
               ):
                  three_in_row_count += 1

      # Check negatively sloped diagonals
      for c in range(column_count - 3):
            for r in range(3, row_count):
               if self.check_winnable(
                  [
                        board[r, c],
                        board[r - 1, c + 1],
                        board[r - 2, c + 2],
                        board[r - 3, c + 3],
                  ],
                  piece,
               ):
                  three_in_row_count += 1

      return three_in_row_count

   def reward(self, done: bool, player: int) -> float:
      """Processes and returns reward from environment according to lesson criteria.

      :param done: Environment has terminated
      :type done: bool
      :param player: Player who we are checking, 0 or 1
      :type player: int
      """
      if done:
            reward = (
               self.lesson["rewards"]["vertical_win"]
               if self.check_vertical_win(player)
               else self.lesson["rewards"]["win"]
            )
      else:
            agent_three_count = self.check_three_in_row(1 - player)
            opp_three_count = self.check_three_in_row(player)
            if (agent_three_count + opp_three_count) == 0:
               reward = self.lesson["rewards"]["play_continues"]
            else:
               reward = (
                  self.lesson["rewards"]["three_in_row"] * agent_three_count
                  + self.lesson["rewards"]["opp_three_in_row"] * opp_three_count
               )
      return reward

   def last(self) -> Tuple[dict, float, bool, bool, dict]:
      """Wrapper around PettingZoo env last method."""
      return self.env.last()

   def step(self, action: int) -> None:
      """Wrapper around PettingZoo env step method."""
      self.env.step(action)

   def reset(self) -> None:
      """Wrapper around PettingZoo env reset method."""
      self.env.reset()