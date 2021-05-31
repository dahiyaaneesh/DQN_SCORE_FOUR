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


layers = [[50],[50,50],[50,50,50],[50,50,50,50]]
activations =['relu', 'tanh']
i = '5_'
dim = (4, 4, 4)

for layer in layers:
    for activation in activations:
        name = "agent"+i+activation+"_"
        tim = strftime("%Y-%m-%d-%H_%M_%S", gmtime())
        callbacks = [TensorBoard(log_dir='runs/run_%s_%s' % (tim, name))]

        env   = ConnectFourEnv(dim=dim, winning_reward=5, losing_reward=-10, tie_reward=0, invalid_move_reward=-100)
        dqn   = make_dqn(env=env, layers=layer, activation = activation)
        agent = InterleavedAgent([dqn, dqn], env=env)
        agent.compile(Adam(lr=1e-3), metrics=['mae'])
        agent.fit(env, nb_steps=500000, visualize=False, verbose=1, callbacks=callbacks)
        agent.save_weights(name, overwrite=True)
        print(i)
    i = '5:'+i

