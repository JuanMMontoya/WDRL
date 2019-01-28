# Wide Deep Reinforcement Learning
Wide Deep Reinforcement Learning (WDRL) implementation in Pac-man using our new algorithm Wide Deep Q-networks (WDQN). 


## Concept

Wide Deep Reinforcement Learning search to combine both the linear function (wide component) and the non-linear function approximation (deep component). 
For our research, we extended the popular Deep Q-Network (DQN) algorithm by mixing the linear combination of features with the convolutional neural networks as illustrated below.
In this way, developing the Wide Deep Q-Network algorithm. 

![](media/WDRL.jpg?raw=true)


## Demo

[![Demo](https://github.com/tychovdo/PacmanDQN/blob/master/videos/PacmanDQN_wingif.gif)](https://youtu.be/QilHGSYbjDQ)

## Replication
To run the models in order to replicate the best results of the study you can call the `runModels.sh` file.
An example of it:


```
$ python3 pacman.py -p PacmanWDQN -n 101 -x 1 -g DirectionalGhost -l mediumClassic --path WDQN-mediumC-3feat --fixRandomSeed 6311
```

Where the program runs on the medium map, evaluating the model for 100 games, using the directional ghoths and the WDQN algoritm with 3 features. In addition, the fix random seed "6111" is used to guarantee replication. To see how this random seed was selected open the file *createRandomSeed.py* 

For replicating 

## Example of how to train

Run a model on `smallGrid` layout for 6000 episodes, of which 5000 episodes
are used for training.

```
$ python3 pacman.py -p PacmanDQN -n 6000 -x 5000 -l smallGrid
```

### Layouts
Different layouts can be found and created in the `layouts` directory.

### Parameters

The hyperparameters used for training the agents can be found in the `dict` folder as `json` files. A complete description of this parameters can be found in the `README` file of the folder.

In the `parameters` folder you can find the additional values needed as `npy` files to properly load the models. <br />
 <br /> 
Models are saved as "checkpoint" files in the `model` directory. <br />
Load and save filenames can be set on the hyperparameters . <br />
 <br />


## Citation

Please cite this repository if it was useful for your research:

```
@inproceedings{montoya2019,
  title={Wide Deep Reinforcement Learning for Grid-Based Action Games},
  author={Montoya, Juan M. and Borgelt, Christian },
  year={2019},
  booktitle = {Proceedings of the Eleventh International Conference on Agents and Artificial Intelligence},
  series = {ICAART'19},
  publisher = {ICAART Press},
}

```

* [Montoya, Juan M. and Borgelt, Christian (2019). Wide Deep Reinforcement Learning for Grid-Based Action Games.](https://notfound.pdf)

## Requirements

- `python==3.5.1`
- `tensorflow==0.8rc`

## Acknoledgements

DQN Framework by  (made for ATARI / Arcade Learning Environment)
* [deepQN_tensorflow](https://github.com/mrkulk/deepQN_tensorflow) ([https://github.com/mrkulk/deepQN_tensorflow](https://github.com/mrkulk/deepQN_tensorflow))

Pac-man implementation by UC Berkeley:
* [The Pac-man Projects - UC Berkeley](http://ai.berkeley.edu/project_overview.html) ([http://ai.berkeley.edu/project_overview.html](http://ai.berkeley.edu/project_overview.html))