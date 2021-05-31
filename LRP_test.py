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
from almost_random import almost_random_player

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

print("For 2D case ! \n\n")
experiments = 500
      

name = "saved_weights/agent_LRP"
dim = (4,4)
env   = ConnectFourEnv(dim=dim, winning_reward=5, losing_reward=-10, tie_reward=0, invalid_move_reward=-100)
dqn   = make_linear_regression(dim=dim)
agent = InterleavedAgent([dqn, dqn], env=env)  
agent.load_weights(name)  
agent.training = False
# Linear agent starts First

step = 0
test_env = ConnectFourEnv(dim=dim, winning_reward=5, losing_reward=-10, tie_reward=0, invalid_move_reward=-100)

done = False
observation = test_env.reset()
agent.training = False
winners =np.zeros(experiments)
count =0
for k in range(len(winners)):
    observation = test_env.reset()
    done = False
    while not done:  
        action = agent.forward(observation)
        observation, reward, done, exp = test_env.step(action)
        if not done:
            action = almost_random_player(test_env, player =-1)
            observation, reward, done, exp = test_env.step(action)
#                 print("Turn: {}".format(step))
        step += 1
        winners[k] = exp['winner']
        if(exp['invalid_move']== True):
            count+=1
#             print("Alert", test_env.current_player)
print("Network won : {} , Network Lost : {}, Network tied : {}".format( sum(winners==1)/len(winners) , sum(winners==-1)/len(winners), sum(winners==0)/len(winners)))
print("Invalid moves made :{}".format(count*1.0/len(winners)))

# Linear agent starts second

step = 0
test_env = ConnectFourEnv(dim=dim, winning_reward=5, losing_reward=-10, tie_reward=0, invalid_move_reward=-100)

done = False
observation = test_env.reset()
agent.training = False
winners =np.zeros(1000)
count =0
for k in range(len(winners)):
    observation = test_env.reset()
    done = False
    while not done:  
        action = almost_random_player(test_env, player =1)
        observation, reward, done, exp = test_env.step(action)
        
        if not done:

            action = agent.forward(observation)
            observation, reward, done, exp = test_env.step(action)
#                 print("Turn: {}".format(step))
        step += 1
        winners[k] = exp['winner']
        if(exp['invalid_move']== True):
            count+=1
#             print("Alert", test_env.current_player)
print("Network won : {} , Network Lost : {}, Network tied : {}".format( sum(winners==-1)/len(winners) , sum(winners==1)/len(winners), sum(winners==0)/len(winners)))
print("Invalid moves made :{}".format(count*1.0/len(winners)))
      
      
print("For 3D case ! \n\n")
      
# Linear agent starts First
dim = (4,4,4)
name = "saved_weights/agent_LRP"
dim = (4,4)
env   = ConnectFourEnv(dim=dim, winning_reward=5, losing_reward=-10, tie_reward=0, invalid_move_reward=-100)
dqn   = make_linear_regression(dim=dim)
agent = InterleavedAgent([dqn, dqn], env=env)  
agent.load_weights(name)  

step = 0
test_env = ConnectFourEnv(dim=dim, winning_reward=5, losing_reward=-10, tie_reward=0, invalid_move_reward=-100)

done = False
observation = test_env.reset()
agent.training = False
winners =np.zeros(experiments)
count =0
for k in range(len(winners)):
    observation = test_env.reset()
    done = False
    while not done:  
        action = agent.forward(observation)
        observation, reward, done, exp = test_env.step(action)
        if not done:
            action = almost_random_player(test_env, player =-1)
            observation, reward, done, exp = test_env.step(action)
#                 print("Turn: {}".format(step))
        step += 1
        winners[k] = exp['winner']
        if(exp['invalid_move']== True):
            count+=1
#             print("Alert", test_env.current_player)
print("Network won : {} , Network Lost : {}, Network tied : {}".format( sum(winners==1)/len(winners) , sum(winners==-1)/len(winners), sum(winners==0)/len(winners)))
print("Invalid moves made :{}".format(count*1.0/len(winners)))

# Linear agent starts second

step = 0
test_env = ConnectFourEnv(dim=dim, winning_reward=5, losing_reward=-10, tie_reward=0, invalid_move_reward=-100)

done = False
observation = test_env.reset()
agent.training = False
winners =np.zeros(experiments)
count =0
for k in range(len(winners)):
    observation = test_env.reset()
    done = False
    while not done:  
        action = almost_random_player(test_env, player =1)
        observation, reward, done, exp = test_env.step(action)
        
        if not done:

            action = agent.forward(observation)
            observation, reward, done, exp = test_env.step(action)
#                 print("Turn: {}".format(step))
        step += 1
        winners[k] = exp['winner']
        if(exp['invalid_move']== True):
            count+=1
#             print("Alert", test_env.current_player)
print("Network won : {} , Network Lost : {}, Network tied : {}".format( sum(winners==-1)/len(winners) , sum(winners==1)/len(winners), sum(winners==0)/len(winners)))
print("Invalid moves made :{}".format(count*1.0/len(winners)))
