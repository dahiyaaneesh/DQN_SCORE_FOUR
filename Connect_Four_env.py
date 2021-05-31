import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gym
import random


class ConnectFourEnv(gym.Env):
    """
    Environment for the connect four game
    Class definition adapted from 
    https://github.com/shakedzy/tic_tac_toe/blob/master/game.py
    Note: In case of invalid move the current player doesn't change and in case of valid move it does.

    Functions
        reset()       : resets the board
        step(action)  : actions of the board. maximum is (dim[0])
        
        
    """
    action_space = gym.spaces.Discrete(4)
    # observation-space = gym.spaces.Discrete(4)
    def __init__(self, dim=(4,4), N=4, winning_reward=10, losing_reward=-10, tie_reward=-5, invalid_move_reward=-100,
                 three_reward=0.6, two_reward=0.4):
        self.dim                 = dim
        self.winning_reward      = winning_reward
        self.losing_reward       = losing_reward
        self.invalid_move_reward = invalid_move_reward
        self.tie_reward          = tie_reward
        self.player1             =  1
        self.player2             = -1
        self.N                   = N

        self.is_2d = len(self.dim) == 2
        self.is_3d = len(self.dim) == 3

        self.reset()

    @property
    def n_actions(self):
        return np.prod(self.dim[:-1])

    @property
    def is_done(self):
        return self._is_done

    def is_full(self):
        return np.sum(self.board == 0) == 0

    def reset(self):
        """
        Resets the board,
        sets the current player as 1, number of steps taken as zero.
        """
        self.board           = np.zeros(shape = self.dim, dtype = int)
        self.current_player  = 1
        self.steps           = 0
        self.invalid_moves   = 0
        self._is_done = False
        return self.board

    def get_action(self, move):
        return np.unravel_index(move, self.dim[:-1])

    def get_move(self, action):
        return np.ravel_multi_index(action, self.dim[:-1])

    def get(self, action=None, height=None, board=None):
        if board is None:
            board = self.board

        if height is None:
            height = slice(None, None)

        if action is not None:
            if self.is_2d:
                return board[action, height].squeeze(0)
            elif self.is_3d:
                return board[action[0], action[1], height]
        else:
            return board[..., height]

    def set(self, action, height, value, board=None):
        if board is None:
            board = self.board

        if self.is_2d:
            board[action, height] = value
        elif self.is_3d:
            board[action[0], action[1], height] = value

    def step(self, move):
        """
        Performs action on the environment corresponding to the current player.
        Input:
            action (int): max size of dim[0]
        """
        if self.is_done:
            raise Exception('You have to reset the game first.')
        action = self.get_action(move)

        # adjust in case too big, could also count as invalid if action >= field_width
        # this is for the human player to make always valid moves
        action = action % self.n_actions

        step_column = self.get(action, self.dim[-1] - 1)
        is_invalid = step_column != 0

        # we should consider to end the game after an invalid move since further play might not be meaningful
        if is_invalid:  # board filled => invalid move.
            info = {'winner': 0, 'game_over': False, 'invalid_move': True}
            # current player is not changed.
            self.invalid_moves += 1
            self._is_done = True
            return self.board, self.invalid_move_reward, self.is_done, info,
        
        else:  # not filled => valid move
            self.steps += 1
            height = self.get_height(action)  # getting the _height at which the token must fall to
            self.set(action, height, value=self.current_player)

            winner = self.check_winner(self.board, action, height)
            if winner != 0:
                info = {'winner': winner, 'game_over': True, 'invalid_move': False}
                reward = self.winning_reward
                self._is_done = True
            else:
                if self.is_full():
                    # Tie
                    info = {'winner': winner, 'game_over': True, 'invalid_move': False}
                    reward = self.tie_reward
                    self._is_done = True
                else:
                    # Proceeding to next turn
                    if self.check_opponent_winning():  # checks whether opponent can win
                        reward = self.losing_reward
                    else:
                        reward = 0
                    info = {'winner': winner, 'game_over': False, 'invalid_move': False}
                    self.current_player *= -1 
            
            return self.board, reward, self.is_done, info

    def _check_winner_3d(self, board, action, height, player=None):
        """
        code yet to be adapted from Till's implementation
        """
        slices = []
        slices.append(board[action[0], :, :])
        slices.append(board[:, action[1], :])
        slices.append(board[:, :, height])
        # todo: stack with a loop for Score N. Also, these slices don't have to be checked all the time, maybe add some if-conditions
        slices.append(np.stack((board[0, 0, :], board[1, 1, :], board[2, 2, :], board[3, 3, :]), axis=0))
        slices.append(np.stack((board[0, 3, :], board[1, 2, :], board[2, 1, :], board[3, 0, :]), axis=0))

        temp = 0
        for slice in slices:
            temp = self.check_combo(slice, player)
            if temp != 0:
                break
        winner = temp

        #game_over = winner != 0 or len(np.argwhere(self.board).reshape(-1, )) == 0
        return winner

    def check_combo(self, matrix, player=None):
        """
        checks for matches along rows, diagonals and columns of a given 2D matrix
        """
        if player is None:
            player = self.current_player
            
        if self.N * player in np.sum(matrix, axis=0):
            return player
        if self.N * player in np.sum(matrix, axis=1):
            return player
        if np.sum(matrix.diagonal()) == self.N * player:
            return player
        if np.sum(np.fliplr(matrix).diagonal()) == self.N * player:
            return player
        return 0
    
    # action == column that is played
    def check_winner(self, board, action, height, player=None):
        """
        checks for the winner in the board, only works for 4,4 case.
        """
        if player is None:
            player = self.current_player
            
        if self.is_2d:
            return self.check_combo(board, player)

        elif self.is_3d:
            return self._check_winner_3d(board, action, height, player)
    
    def check_opponent_winning(self):
        """
        Checks if the opponent can win next game or not.
        """
        valid_actions = self.get_valid_actions()
        copy_board = np.copy(self.board)
        for action in list(valid_actions):
            height = self.get_height(action, board=copy_board)
            self.set(action, height=height, value=self.current_player * -1, board=copy_board)

            if self.check_winner(copy_board, action, height) != 0:
                return True

            self.set(action, height=height, value=0, board=copy_board)

        return False

    def get_height(self, action, board=None):
        return np.argwhere(self.get(action=action, board=board) == 0)[0, 0]

    def get_valid_actions(self, board=None):
        return np.argwhere(self.get(height=self.dim[-1] - 1, board=board) == 0)

    def render(self, mode = 'human'):
        """
        For displaying the game to human.
        """
        if mode == 'human':

            if self.is_2d:
                fig = plt.figure()
                for index, value in np.ndenumerate(self.board):
                    if value == 1:
                        plt.scatter(*index, c='red', s=1000, alpha=0.2)
                    elif value == -1:
                        plt.scatter(*index, c='blue', s=1000, alpha=0.2)
                plt.xlim(-1, self.dim[0])
                plt.ylim(-1, self.dim[1])
                plt.xticks([])
                plt.yticks([])
                plt.grid(True)

            if self.is_3d:
                fig = plt.figure()
                ax = Axes3D(fig)
                for index, value in np.ndenumerate(self.board):
                    if value == 1:
                        ax.scatter(*index, c='red', s=1000, alpha=0.2)
                    elif value == -1:
                        ax.scatter(*index, c='blue', s=1000, alpha=0.2)
                ax.set_xlim(0, self.dim[0] - 1)
                ax.set_ylim(0, self.dim[1] - 1)
                ax.set_zlim(0, self.dim[2] - 1)

        else:
            fig = plt.figure()
            ax = Axes3D(fig)
            for index, value in np.ndenumerate(self.board):
                if value == 1:
                    ax.scatter(*index, c='red', s=1000, alpha=0.2)
                elif value == -1:
                    ax.scatter(*index, c='blue', s=1000, alpha=0.2)

            ax.set_xlim(0, self.dim[0] - 1)
            ax.set_ylim(0, self.dim[1] - 1)
            ax.set_zlim(0, self.dim[2] - 1)
            ax.set_title('Nr of steps: ' + str(self.steps))

        plt.show()
        return fig
