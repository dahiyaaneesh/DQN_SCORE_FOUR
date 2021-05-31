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

from random import sample 
from almost_random import almost_random_player

print("ARP starts plays first")
# When the almost random player plays first
layers = [[50],[50,50],[50,50,50],[50,50,50,50]]
activations =['relu', 'tanh']
experiments = 2
dim=(4,4)
for activation in activations:
    i = '5_'
    for layer in layers:
        count = 0
        test_env = ConnectFourEnv(dim=dim,winning_reward=5, losing_reward=-10, tie_reward=0, invalid_move_reward=-100)
        dqn = make_dqn(layers=layer, activation=activation, env=test_env)
        agent = InterleavedAgent([dqn, dqn], env=test_env)

        agent.load_weights("saved_weights/agent"+i+activation+"_")
        step = 0

        done = False
        observation = test_env.reset()
        agent.training = False
        winners =np.zeros(experiments)
        for k in range(len(winners)):
            observation = test_env.reset()
            done = False
            while not done:  
                action = almost_random_player(test_env, player=1)
                observation, reward, done, exp = test_env.step(action)
                if not done:
                    action = agent.forward(observation)
                    observation, reward, done, exp = test_env.step(action)
#                 print("Turn: {}".format(step))
                step += 1
                winners[k] = exp['winner']
                if(exp['invalid_move']== True):
                    count+=1
#                     print("Alert", test_env.current_player)
        print("Layers : {}, Activation: {},  Network won : {} , Network lost : {}, Network tied : {}".format(
            i,activation, sum(winners==-1)/len(winners) , sum(winners==1)/len(winners), sum(winners==0)/len(winners)))
        print("Invalid_moves {}\n\n".format(count))        
#         print(i)
        i = '5:'+i

# When the almost random player plays second
# todo: duplicated code, try to do that in one loop maybe
layers = [[50],[50,50],[50,50,50],[50,50,50,50]]
activations =['relu', 'tanh']
i = '5_'
print("Network starts")
for activation in activations:
    i = '5_'
    for layer in layers:
        test_env = ConnectFourEnv(dim=dim, winning_reward=5, losing_reward=-10, tie_reward=0, invalid_move_reward=-100)
        dqn = make_dqn(layers=layer, activation=activation, env=test_env)
        agent = InterleavedAgent([dqn, dqn], env=test_env)
        agent.load_weights("saved_weights/agent"+i+activation+"_")
        step = 0

        done = False
        observation = test_env.reset()
        agent.training = False
        winners =np.zeros(experiments)
        count = 0
        for k in range(len(winners)):
            observation = test_env.reset()
            done = False
            while not done:  
                action = agent.forward(observation)
                observation, reward, done, exp = test_env.step(action)
                if not done:
                    action = almost_random_player(test_env, player=-1)
                    observation, reward, done, exp = test_env.step(action)
#                 print("Turn: {}".format(step))
                step += 1
                winners[k] = exp['winner']
                if(exp['invalid_move']== True):
                    count+=1
#                     print("Alert", test_env.current_player)
        print("Layers : {}, Activation: {},  Network won : {} , Network lost : {}, Network tied : {}".format(
            i,activation, sum(winners==1)/len(winners) , sum(winners==-1)/len(winners), sum(winners==0)/len(winners)))
        print("Invalid_moves {}\n\n".format(count))       
        # print(i)
        i = '5:'+i