# Parameters Folder #

The npy files save the parameters necessary for loading DQN, Linear and WDQN agents.

The following variables are saved in a `npy` file:

Total amount of steps until the previous episode : `last_steps` <br />
Total amount of rewards accumulated during training: `accumTrainRewards` <br />
Saves the lastest hyperparameters selected : `params` <br />
Local count of the actual episode: `local_cnt` <br />
Total amount of steps/counts for all episodes : `cnt` <br />
****