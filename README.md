# Wide Deep Reinforcement Learning
Wide Deep Reinforcement Learning (WDRL) in Pac-man with the new algorithm Wide Deep Q-networks (WDQN). 


## Concept

Wide Deep Reinforcement Learning search to combine both the linear function (wide component) and the non-linear function approximation (deep component). 
For our research, we extended the popular Deep Q-Network (DQN) algorithm by mixing the linear combination of features with the convolutional neural networks as illustrated below.
In this way, developing the Wide Deep Q-Network algorithm. 
<object data="https://github.com/JuanMMontoya/WDRL/media/WDRL.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="https://github.com/JuanMMontoya/WDRL/media/WDRL.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="https://github.com/JuanMMontoya/WDRL/blob/master/media/WDRL.pdf">Download PDF</a>.</p>
    </embed>
</object>

## Demo

[![Demo](https://github.com/tychovdo/PacmanDQN/blob/master/videos/PacmanDQN_wingif.gif)](https://youtu.be/QilHGSYbjDQ)

## Run models
To run the models for replicating the study you can call the *runModels.sh* file.
An example of it: 


```
$ python3 pacman.py -p PacmanWDQN -n 101 -x 1 -g DirectionalGhost -l mediumClassic --path WDQN-mediumC-3feat --fixRandomSeed 6311
```

Where the program runs on the medium map, evaluating the model for 100 games, using the directional ghoths and the WDQN algoritm with 3 features. In addition, the fix random seed "6111" is used to guarantee replication. To see how this random seed was selected open the file *createRandomSeed.py* 


## Example of how to train

Run a model on `smallGrid` layout for 6000 episodes, of which 5000 episodes
are used for training.

```
$ python3 pacman.py -p PacmanDQN -n 6000 -x 5000 -l smallGrid
```

### Layouts
Different layouts can be found and created in the `layouts` directory

### Parameters

Parameters can be found in the `params` dictionary in `pacmanDQN_Agents.py`. <br />
 <br />
Models are saved as "checkpoint" files in the `/saves` directory. <br />
Load and save filenames can be set using the `load_file` and `save_file` parameters. <br />
 <br />
Episodes before training starts: `train_start` <br />
Size of replay memory batch size: `batch_size` <br />
Amount of experience tuples in replay memory: `mem_size` <br />
Discount rate (gamma value): `discount` <br />
Learning rate: `lr` <br />
 <br />
Exploration/Exploitation (ε-greedy): <br />
Epsilon start value: `eps` <br />
Epsilon final value: `eps_final` <br />
Number of steps between start and final epsilon value (linear): `eps_step` <br />

## Citation

Please cite this repository if it was useful for your research:

```
@article{van2016deep,
  title={Deep Reinforcement Learning in Pac-man},
  subtitle={Bachelor Thesis},
  author={van der Ouderaa, Tycho},
  year={2016},
  school={University of Amsterdam},
  type={Bachelor Thesis},
  pdf={https://esc.fnwi.uva.nl/thesis/centraal/files/f323981448.pdf},
}

```

* [van der Ouderaa, Tycho (2016). Deep Reinforcement Learning in Pac-man.](https://esc.fnwi.uva.nl/thesis/centraal/files/f323981448.pdf)

## Requirements

- `python==3.5.1`
- `tensorflow==0.8rc`

## Acknoledgements

DQN Framework by  (made for ATARI / Arcade Learning Environment)
* [deepQN_tensorflow](https://github.com/mrkulk/deepQN_tensorflow) ([https://github.com/mrkulk/deepQN_tensorflow](https://github.com/mrkulk/deepQN_tensorflow))

Pac-man implementation by UC Berkeley:
* [The Pac-man Projects - UC Berkeley](http://ai.berkeley.edu/project_overview.html) ([http://ai.berkeley.edu/project_overview.html](http://ai.berkeley.edu/project_overview.html))