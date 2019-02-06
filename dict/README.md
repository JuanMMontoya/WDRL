# DICT Folder

This part has the json files containing the parameters for initializing the DQN, Linear and WDQN agents.

The abbreviations have the following meaning: <br />

Episodes before training starts: `train_start` <br />
Size of replay memory batch size: `batch_size` <br />
Amount of experience tuples in replay memory: `mem_size` <br />
Discount rate (gamma value): `discount` <br />
Update target network after n amount of episodes: `target_update_network` <br />
The probability that each element is kept: `dropout` <br />
Counts of steps of linear model (default nUll): `global_step_lin` <br />
Counts of steps of DQN model (default null): `global_step_dqn` <br />
**Learning rate:** <br />
DQN learning rate: `lr_dqn` <br />
Linear learning rate: `lr_lin`<br />
**Exponential decay for the linear learning rate: <br />**
Boolean for turning off or on the exponential decay: `dcy_lrl` <br />
Dictionary with decay_steps and decay_rate: `dcy_lrl_val` <br />
**Exploration/Exploitation (ε-greedy):** <br />
Epsilon start value: `eps` <br />
Epsilon final value: `eps_final` <br />
Number of steps between start and final epsilon value (linear): `eps_step` <br />
**Saving/Loading:** <br />
Name of memory replay to load: `load_file` <br />
Boolean for saving model and memory replay file: `save` <br />
Boolean for loading model file: `load` <br />
Boolean for loading the memory replay: `load_data` <br />
Interval to save: `save_interval` <br />
Boolean for saving logs: `save_logs` <br />
Threshold to determine when to save the best model: `best_thr` <br />
**RMSPropOptimizer (for testing):** <br />
Discounting factor for the history/coming gradient: `rms_decay` <br />
Small value to avoid zero denominator:`rms_eps` <br />
**Neural Network Architecture and Linear Combination of features:** <br />
Selects N amount of convolutional layers: `conv_layer_sizes` <br />
Selects N amount of fully connected layers: `hidden_layer_sizes` <br />
Number of outputs in the model: `k` <br />
Number of inputs in the model: `mat_dim` <br />
Selected features to be used in the linear model: `feat_val` <br />
**Type of Agent** <br />
Selects the linear agent: `only_lin` <br />
Selects the DQN agent:`only_dqn` <br />
**Remark:** If both `only_lin` and `only_dqn` are true or false the WDQN Agent will be automatically selected.
