# Deep Q-Network for score four and connect four.

### Abstract
**In this report, we aim to explore how the complexity of hidden layers affects the learning capacity of an agent in the context of reinforcement learning. We trained a deep neural network to approximate the Q-function of the popular two-player games Connect Four and Score Four. We vary the capacity by changing the number of hidden layers and the corresponding activation function. To evaluate the performance, we picked several test cases and observed how the agent performs. As baseline, we implemented a player with a few hard-coded rules. We compared this player with the trained networks and with a player trained on the same objective,but with linear regression, which served as another baseline.We found that the neural networks outperform the linear regression player compared to the hard-coded player. Our results concerning the network architecture are inconclusive.**
*Full report is in /report.pdf*

Authors:
1. Aneesh Dahiya
2. Jim Buffat
3. Jonathan Lehner
4. Till Schnabel

**All auhtors contributed equally to this work.**

## Requirements:
These were the packages and module versions we used during our experiments. 
1. python==3.5
2. tensorflow==1.14.0
3. keras==2.3.1
4. keras-rl==0.4.2
Apart from these all other packages come with the standard distribution of python. The code can be run on Google's colaboratory as well provided these packages are downloaded in the environment.
The code doesn't work with any other tensorflow version

## Files description:
### Scripts
1. Agent.py: Contains the class definition for "InterleavedAgent". 
2. network.py : Contains the function "make_dqn" for defining the neural network used by the InterleavedAgent class.
3. Connect_Four_env.py : contains the "ConnectFourEnv" class for score four and connect four games.
4. almost_random.py : Contains the function for AlmostRandomPlayer (ARP).
### Scripts for training:
To run :
>python <name_of_the_Script>.py to run.

1. train_2d.py : For training the agents for Connect four.
2. train_3d.py : For training the agents for Score four. 
3. LRP_train.py   : Trains LRP for both score four and connect four games.

###Scipts for model evaluation:
1. ARP_test_2d.py : For testing agents against ARP for Connect Four.
2. ARP_test_3d.py : For testing agents against ARP for Score Four.
3. LRP_test.py : For testing LRP against ARP for Score Four

### Directories:
saved_weights : Directory containing saved weights.
runs          : Directory conatining files for plotting plots on tensorboard.



> For references please refer to report.
## Endnotes
1. These scripts are part of the project of Deep Learning course held in Autumn2019-20 at ETH Zurich. Please read the report for more info.
2. You may retrain the networks to generate new weights by running scripts for traininig. 
3. Training takes considerable amount of time, upwards of an hour.
4. Experiments can be performed again with scripts for model evaluation


