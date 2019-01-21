# Pick a Random Number for testing the efficiency of models
import random as r
import os
import _pickle as cPickle

r.seed(0)
magic_random = r.randint(0, 10000)

print("The random seed number is", magic_random)

r.seed(magic_random)
l_rand = [i for i in range(1, 100001)]
r.shuffle(l_rand)

with open("".join(["random/", "randomList", ".pkl"]), "wb") as file:
    cPickle.dump(l_rand, file, protocol=3)
