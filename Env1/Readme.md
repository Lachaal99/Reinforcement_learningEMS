## Reinforcement learning for EMS 
This repository contains the first implementation of an rl model for EMS :
    - Algorithms : Q learning, DQN, MPC ,SAC
    - Environment : customized one based on models
    - Discrete state and action space for q learning
    - Continuous space state and discrete action space DQN
    - Continous space and space action for both the model predictive control and Soft actor critic

1h step was deployed.
repository structure : 
For each algorithm , you will find a python file containing the environment (customized to the deployed algorithm) along with the agent build up.
each file can be executed to train the model. adding for that you will find 2 seperate files for sac and DQN inference as these are NN based algorithms , so the model is saved for later inference after training.