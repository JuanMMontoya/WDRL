#!/bin/bash
#WARNING: For replicating study put the json files
# into dict/mediumC and dict/smallC respectively.
#MEDIUM MAP

#DQN
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path DQN-mediumC-a --fixRandomSeed 1
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path DQN-mediumC-b --fixRandomSeed 2
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path DQN-mediumC-c --fixRandomSeed 3
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path DQN-mediumC-d --fixRandomSeed 4
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path DQN-mediumC-e --fixRandomSeed 5

#WDQN 3 feat
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path WDQN-mediumC-feat3-a --fixRandomSeed 1
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path WDQN-mediumC-feat3-b --fixRandomSeed 2
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path WDQN-mediumC-feat3-c --fixRandomSeed 3
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path WDQN-mediumC-feat3-d --fixRandomSeed 4
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path WDQN-mediumC-feat3-e --fixRandomSeed 5

#WDQN 2 feat
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path WDQN-mediumC-feat2-a --fixRandomSeed 1
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path WDQN-mediumC-feat2-b --fixRandomSeed 2
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path WDQN-mediumC-feat2-c --fixRandomSeed 3
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path WDQN-mediumC-feat2-d --fixRandomSeed 4
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path WDQN-mediumC-feat2-e --fixRandomSeed 5

#WDQN 1 feat
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path WDQN-mediumC-feat1-a --fixRandomSeed 1
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path WDQN-mediumC-feat1-b --fixRandomSeed 2
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path WDQN-mediumC-feat1-c --fixRandomSeed 3
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path WDQN-mediumC-feat1-d --fixRandomSeed 4
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path WDQN-mediumC-feat1-e --fixRandomSeed 5

#Lin
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path LIN-mediumC-a --fixRandomSeed 1
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path LIN-mediumC-b --fixRandomSeed 2
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path LIN-mediumC-c --fixRandomSeed 3
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path LIN-mediumC-d --fixRandomSeed 4
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path LIN-mediumC-e --fixRandomSeed 5


#SMALL MAP

#DQN
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path DQN-smallC-a --fixRandomSeed 1
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path DQN-smallC-b --fixRandomSeed 2
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path DQN-smallC-c --fixRandomSeed 3
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path DQN-smallC-d --fixRandomSeed 4
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path DQN-smallC-e --fixRandomSeed 5

#WDQN 3 feat
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path WDQN-smallC-feat3-a --fixRandomSeed 1
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path WDQN-smallC-feat3-b --fixRandomSeed 2
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path WDQN-smallC-feat3-c --fixRandomSeed 3
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path WDQN-smallC-feat3-d --fixRandomSeed 4
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path WDQN-smallC-feat3-e --fixRandomSeed 5

#WDQN 2 feat
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path WDQN-smallC-feat2-a --fixRandomSeed 1
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path WDQN-smallC-feat2-b --fixRandomSeed 2
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path WDQN-smallC-feat2-c --fixRandomSeed 3
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path WDQN-smallC-feat2-d --fixRandomSeed 4
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path WDQN-smallC-feat2-e --fixRandomSeed 5

#WDQN 1 feat
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path WDQN-smallC-feat1-a --fixRandomSeed 1
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path WDQN-smallC-feat1-b --fixRandomSeed 2
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path WDQN-smallC-feat1-c --fixRandomSeed 3
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path WDQN-smallC-feat1-d --fixRandomSeed 4
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path WDQN-smallC-feat1-e --fixRandomSeed 5

#LIN
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path LIN-smallC-a --fixRandomSeed 1
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path LIN-smallC-b --fixRandomSeed 2
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path LIN-smallC-c --fixRandomSeed 3
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path LIN-smallC-d --fixRandomSeed 4
python3 pacman.py -p PacmanWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path LIN-smallC-e --fixRandomSeed 5
