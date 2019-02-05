# DICT Folder

This part has the json files containing the parameters for initializing the DQN, Linear and WDQN agents.

The abbreviations have the following meaning: <br />

Episodes before training starts: `train_start` <br />
Size of replay memory batch size: `batch_size` <br />
Amount of experience tuples in replay memory: `mem_size` <br />
Discount rate (gamma value): `discount` <br />
Learning rate: <br />
DQN learning rate: `lr_dqn` <br />
Linear learning rate: `lr_lin`<br />
Exponential decay for the linear learning rate: <br />
Boolean for turning off or on the exponential decay: `dcy_lrl` <br />
Dictionary with decay_steps and decay_rate: `dcy_lrl_val` <br />
Exploration/Exploitation (ε-greedy): <br />
Epsilon start value: `eps` <br />
Epsilon final value: `eps_final` <br />
Number of steps between start and final epsilon value (linear): `eps_step` <br />



`load_file` <br />
`save` <br />
`load` <br />
`load_data` <br />
`save_interval` <br />
`global_step_lin` <br />
`global_step_dqn` <br />
`save_logs` <br />
`target_update_network` <br />
`dropout` <br />
`best_thr` <br />

`rms_decay` <br />
`rms_eps` <br />
`conv_layer_sizes` <br />
`hidden_layer_sizes` <br />

`k` <br />
`mat_dim`
`feat_val` <br />
`only_lin` <br />
`only_dqn` <br />
