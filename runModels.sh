# Running best models after training
# MEDIUM MAP
# WDQN 3 feat
python3 pacman.py -p PacmanWDQN -n 101 -x 1 -g DirectionalGhost -l mediumClassic --path WDQN-mediumC-3feat --fixRandomSeed 6311
# WDQN 2 feat
python3 pacman.py -p PacmanWDQN -n 101 -x 1 -g DirectionalGhost -l mediumClassic --path WDQN-mediumC-2feat --fixRandomSeed 6311
# WDQN 1 feat
python3 pacman.py -p PacmanWDQN -n 101 -x 1 -g DirectionalGhost -l mediumClassic --path WDQN-mediumC-1feat --fixRandomSeed 6311
# DQN
python3 pacman.py -p PacmanWDQN -n 101 -x 1 -g DirectionalGhost -l mediumClassic --path DQN-mediumC --fixRandomSeed 6311
# Linear
python3 pacman.py -p PacmanWDQN -n 101 -x 1 -g DirectionalGhost -l mediumClassic --path LIN-mediumC --fixRandomSeed 6311

# SMALL MAP
# WDQN 3 feat
python3 pacman.py -p PacmanWDQN -n 101 -x 1 -g DirectionalGhost -l smallClassic --path WDQN-smallC-3feat --fixRandomSeed 6311
# WDQN 2 feat
python3 pacman.py -p PacmanWDQN -n 101 -x 1 -g DirectionalGhost -l smallClassic --path WDQN-smallC-2feat --fixRandomSeed 6311
# WDQN 1 feat
python3 pacman.py -p PacmanWDQN -n 101 -x 1 -g DirectionalGhost -l smallClassic --path WDQN-smallC-1feat --fixRandomSeed 6311
# DQN
python3 pacman.py -p PacmanWDQN -n 101 -x 1 -g DirectionalGhost -l smallClassic --path DQN-smallC --fixRandomSeed 6311