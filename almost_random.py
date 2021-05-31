from Connect_Four_env import ConnectFourEnv
from Agent import InterleavedAgent
from network import make_dqn

from random import sample
import numpy as np


def almost_random_player(env, player=-1):
    """
    Makes an almost random move if it doesn't result in victory of the oppoenent.
    player (int) : player id of the almost_random_player
    env    (object): Environment object of the Connect_four class.
    """

    board = np.copy(env.board)
    list_of_moves = []
    env2 = ConnectFourEnv(dim=env.dim)
    env2.reset()
    env2.board = np.copy(board)
    env2.current_player = player
    if(len(env.dim)==2):
    	valid_actions = list(env2.get_valid_actions(env2.board))
    else:
    	valid_actions = env2.get_valid_actions(env2.board)
    assert(len(valid_actions) >= 1)
    for action in valid_actions:
        env2.reset()
        env2.board = np.copy(board)
        env2.current_player = player
        move = env2.get_move(action)
        _, reward, done, info = env2.step(move)
        if info['winner'] == player:
            return move
        valid_opp_actions = env2.get_valid_actions(env2.board)
        if action in valid_opp_actions:
            _, reward, done, info = env2.step(move)
            if info['winner'] != (-1)*player:
                list_of_moves.append(move)
    for action in valid_actions:
        env2.reset()
        env2.board = np.copy(board)
        env2.current_player = (-1)*player
        move = env2.get_move(action)
        _, reward, done, info = env2.step(move)
        if info['winner'] == (-1)*player:
            return move
    if len(list_of_moves) > 0:
        return sample(list_of_moves, 1)[0]
    else:
        return sample(valid_actions, 1)[0]
