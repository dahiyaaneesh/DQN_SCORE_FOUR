from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Embedding
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent 
from rl.policy import EpsGreedyQPolicy 
from rl.memory import SequentialMemory 


def make_dqn(env, layers = [50,50,50,50],activation = "tanh", eps = 0.5):
    """
    Creates a networ with first layer as tanh, then len(layers) hidden layers,
    followed by the output layer with nb_actions as number of units
    """
    model = Sequential()
    # input layer.
    model.add(Flatten(input_shape=(1,) + env.dim))
    model.add(Activation('tanh'))
    # Hidden layers.
    for units in layers: 
        model.add(Dense(units))
        model.add(Activation(activation))
    # output layers.    
    model.add(Dense(env.n_actions))
    model.add(Activation('linear'))
    
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = EpsGreedyQPolicy(eps=eps)
    dqn = DQNAgent(model=model, nb_actions=env.n_actions, memory=memory, nb_steps_warmup=100, target_model_update=1e-2,
                   policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    
    return dqn
