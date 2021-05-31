import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gym
import random
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Embedding
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent 
from rl.policy import EpsGreedyQPolicy 
from rl.memory import SequentialMemory


from Connect_Four_env import ConnectFourEnv
from Agent import InterleavedAgent
from network import make_dqn

from keras.callbacks import TensorBoard
from time import strftime, gmtime

def make_linear_regression(dim =(4,4),eps = 0.5, nb_action = 4):
    """
    Creates a networ with first layer as tanh, then len(layers) hidden layers,
    followed by the output layer with nb_actions as number of units
    """
    nb_actions = 4
    model = Sequential()
    # input layer.
    model.add(Flatten(input_shape=(1,) +dim))
    model.add(Activation('linear'))
    # output layers.    
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = EpsGreedyQPolicy(eps=eps)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100, target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    
    return dqn

print("------Trainig for 3d case!------\n\n\n")
dim = (4,4,4)
name = "agent_LRP"
tim = strftime("%Y-%m-%d-%H_%M_%S", gmtime())
callbacks = [TensorBoard(log_dir='runs/run_%s_%s' % (tim, name))]

env   = ConnectFourEnv(dim=dim, winning_reward=5, losing_reward=-10, tie_reward=0, invalid_move_reward=-100)
dqn   = make_linear_regression(dim=dim)
# arp = AlmostRandomPlayer(env)
agent = InterleavedAgent([dqn, dqn], env=env)
agent.compile(Adam(lr=1e-3), metrics=['mae'])
agent.fit(env, nb_steps=500000, visualize=False, verbose=1, callbacks=callbacks)
agent.save_weights(name, overwrite=True)

print("------Trainig for 2d case!------\n\n\n")
dim = (4,4)
name = "agent_LRP"
tim = strftime("%Y-%m-%d-%H_%M_%S", gmtime())
callbacks = [TensorBoard(log_dir='runs/run_%s_%s' % (tim, name))]

env   = ConnectFourEnv(dim=dim, winning_reward=5, losing_reward=-10, tie_reward=0, invalid_move_reward=-100)
dqn   = make_linear_regression(dim=dim)
# arp = AlmostRandomPlayer(env)
agent = InterleavedAgent([dqn, dqn], env=env)
agent.compile(Adam(lr=1e-3), metrics=['mae'])
agent.fit(env, nb_steps=500000, visualize=False, verbose=1, callbacks=callbacks)
agent.save_weights(name, overwrite=True)


