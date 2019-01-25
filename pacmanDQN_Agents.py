"""
Author: Juan M. Montoya
Code Based on
PacmanDQN implementation by Tycho van der Ouderaa found at
https://github.com/tychovdo/PacmanDQN/

The Pacman AI projects were developed at UC Berkeley found at
http://ai.berkeley.edu/project_overview.html
"""
from util import *
import time
import sys
import numpy as np

# Pacman game
from game import Agent
from pacman import GameState

# Neural nets
import tensorflow as tf
from DQN import DQN


class PacmanDQN(Agent):
    """
    Creates the Deep Q-Network Agent that iterates with the environment
    """

    def __init__(self, args):
        print("Initialise DQN Agent")
        # Load parameters from user-given arguments
        self.params = json_to_dict(args["path"])
        self.params['width'] = args['width']
        self.params['height'] = args['height']
        self.params['num_training'] = args['numTraining']
        self.params['num_games'] = args['numGames']
        self.path_extra = ""
        self.params["seed"] = args['seed']
        self.random = np.random.RandomState(self.params["seed"])
        print(self.params["save_file"])
        print("seed", self.params["seed"])

        # Start Tensorflow session
        tf.reset_default_graph()
        tf.set_random_seed(self.params["seed"])
        self.qnet = DQN(self.params, "model")  # Q-network
        self.tnet = DQN(self.params, "target_model")  # Q-target-network
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.qnet.set_session(self.sess)
        self.tnet.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

        # Time started
        self.general_record_time = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime())

        # Q and cost
        self.Q_global = []
        self.cost_disp = 0

        # Stats
        self.cnt = self.qnet.sess.run(self.qnet.global_step)
        self.local_cnt = 10000
        self.wins = 0
        self.best_int = 100
        self.numeps = 0
        self.episodeStartTime = time.time()
        self.last_steps = 0
        self.get_direction = lambda k: ['North', 'South', 'East', 'West', 'Stop'][k]
        self.get_value = {'North': 0, 'South': 1, 'East': 2, 'West': 3, 'Stop': 4}
        self.lastWindowAccumRewards = 0.0
        self.Q_accumulative = 0.0
        self.accumTrainRewards = 0.0
        self.sub_dir = str(self.params["save_interval"])

    def registerInitialState(self, state):
        """Inspects the starting state"""
        # Reset reward
        self.last_score = 0
        self.last_reward = 0.

        # Reset state
        self.last_state = None
        self.current_state = state

        # Reset actions
        self.last_action = None

        # Reset vars
        self.terminal = None
        self.won = True
        self.Q_global = []

        # Load model
        self.load_mod()

        # Next
        self.numeps += 1

    def getQvalues(self, model, dropout):
        """Access Q Values by using the model prediction of DQN.py"""
        return model.predict(map_state_mat(self.current_state), dropout)[0]

    def getPolicy(self, model, dropout=1.0):
        """Pick up the policy """
        qValues = self.getQvalues(model, dropout)
        qVal = {self.get_value[l]: qValues[self.get_value[l]] for l in self.current_state.getLegalActions(0) if
                not l == "Stop"}
        maxValue = max(qVal.values())
        self.Q_global.append(maxValue)
        return self.get_direction(self.random.choice([k for k in qVal.keys() if qVal[k] == maxValue]))

    def getAction(self, state):
        """Exploit / Explore"""
        if self.random.rand() > self.params['eps']:
            # Exploit action
            move = self.getPolicy(self.qnet)  # dropout deactivated
        else:
            legal = [v for v in state.getLegalActions(0) if not v == "Stop"]
            move = self.random.choice(legal)
        # Save last_action
        self.last_action = self.get_value[move]
        return move

    def observationFunction(self, state):
        """Do observation"""
        self.terminal = False
        self.observation_step(state)
        return state

    def observation_step(self, state):
        """
        Realize the observation step
        Rewards are balanced in this part
        The training occurs in this section
        """
        if self.last_action is not None:
            # Process current experience state
            self.last_state = self.current_state.deepCopy()
            self.current_state = state
            # Process current experience reward
            reward = state.getScore() - self.last_score
            self.last_score = state.getScore()
            # Reward system
            if reward > 20:
                self.last_reward = 50.  # Eat ghost
            elif reward > 0:
                self.last_reward = 10.  # Eat food
            elif reward < -10:
                self.last_reward = -500.  # Get eaten
                self.won = False
            elif reward < 0:
                self.last_reward = -1.  # Punish time

            if (self.terminal and self.won):
                self.last_reward = 100.

            if self.isInTraining():
                if self.local_cnt % self.params["target_update_network"] == 0 and \
                        self.local_cnt > self.params['train_start']:
                    self.tnet.rep_network(self.qnet)
                    print("Copied model parameters to target network. total_t = %s, period = %s" % (
                        self.local_cnt, self.params["target_update_network"]))
                # Store last experience into memory
                experience = (
                    self.last_state, float(self.last_reward), self.last_action, self.current_state, self.terminal)

                if self.params["pickle"]:
                    save_mem_rep_pkl(experience, self.params["save_file"], self.sub_dir, self.local_cnt,
                                     self.params["mem_size"], self.path_extra)
                else:
                    save_mem_rep(experience, self.params["save_file"], self.sub_dir, self.local_cnt,
                                 self.params["mem_size"], self.path_extra)

                # Train
                self.train()

                # Next and Reset numeps when starting to train
                self.local_cnt += 1
                if self.local_cnt == self.params['train_start']:
                    print("")
                    print("######Memory Replay populated########")
                    print("")

                self.params['eps'] = max(self.params['eps_final'],
                                         1.00 - float(self.cnt) / float(self.params['eps_step']))

    def train(self):
        """Train"""
        if self.local_cnt > self.params['train_start']:
            batch_s, batch_a, batch_t, q_t, batch_r = extract_batches_dqn(self.params, self.local_cnt, self.tnet,
                                                                          self.path_extra, self.sub_dir, GameState,
                                                                          self.random)
            self.cnt, self.cost_disp = self.qnet.train(batch_s, batch_a, batch_t, q_t, batch_r, self.params["dropout"])

    def final(self, state):
        """Inspects the last state"""
        self.terminal = True
        self.observation_step(state)
        NUM_EPS_UPDATE = 100
        self.lastWindowAccumRewards += state.getScore()
        self.accumTrainRewards += state.getScore()
        self.Q_accumulative += max(self.Q_global, default=float('nan'))
        self.wins += self.won
        if self.numeps % NUM_EPS_UPDATE == 0:
            eps_time = time.time() - self.episodeStartTime
            print('Reinforcement Learning Status:')
            if self.numeps <= self.params['num_training']:
                trainAvg = self.accumTrainRewards / float(self.numeps)
                print('\tCompleted %d out of %d training episodes' % (
                    self.numeps, self.params['num_training']))
                print('\tAverage Rewards over all training: %.2f' % (
                    trainAvg))

            windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
            windowQavg = self.Q_accumulative / float(NUM_EPS_UPDATE)
            window_steps = (self.cnt - self.last_steps) / float(NUM_EPS_UPDATE)
            print('\tAverage Rewards for last %d episodes: %.2f' % (
                NUM_EPS_UPDATE, windowAvg))
            print('\tEpisode took %.2f seconds' % (eps_time))
            print('\tEpisilon is %.8f' % self.params["eps"])

            if self.params["save_logs"]:
                log_file = open('logs/' + self.params["save_file"] + "-" + str(self.general_record_time) + '-l-' + str(
                    self.params["num_training"]) + '.log', 'a')
                log_file.write("# %4d |  s: %8d | t: %.2f  |  r: %12f | Q: %10f | won: %r \n" %
                               (self.numeps, window_steps, eps_time, windowAvg,
                                windowQavg, self.wins))

            # Save Best Model
            if windowAvg >= self.params["best_thr"]:
                self.params["best_thr"] = windowAvg
                self.last_steps = self.cnt
                self.save_mod(best_mod=True)
                print("Saving the model with:", self.params["best_thr"])
                self.params["best_thr"] = self.params["best_thr"] + self.best_int

            sys.stdout.flush()
            self.lastWindowAccumRewards = 0
            self.Q_accumulative = 0
            self.last_steps = self.cnt
            self.wins = 0
            self.episodeStartTime = time.time()

        if self.numeps >= self.params['num_training']:
            print('Training Done (turning off epsilon)')
            eps_time = time.time() - self.episodeStartTime
            self.params["eps"] = 0.0  # no exploration
            log_file = open(
                'logs/testedModels/' + self.params["save_file"] + '-s-' + str(self.params["seed"]) + '-n-' + str(
                    self.params['num_games'] - self.params["num_training"]) + '.log',
                'a')
            log_file.write("# %4d |  s: %8d | t: %.2f  |  r: %12f | Q: %10f | won: %r \n" %
                           (self.numeps, self.cnt - self.last_steps, eps_time, state.getScore(),
                            max(self.Q_global, default=float('nan'))
                            , int(self.won)))
            self.last_steps = self.cnt

        # Save model
        self.save_mod(best_mod=False)

    def isInTraining(self):
        """Check is if agent is in training"""
        return self.numeps < self.params["num_training"]

    def isInTesting(self):
        """Check is if agent is in testing"""
        return not self.isInTraining()

    def load_mod(self):
        """ Load data and model"""
        if self.params["load"]:
            try:
                self.saver.restore(self.sess,
                                   "".join([self.path_extra, "model/", self.params["save_file"], "-",
                                            self.params["load_file"]]))
                if not self.params["load_file"].lower() == "best":
                    print("Model Restored")
                else:
                    print("Best Model Restored")
                try:
                    load_path = "".join(
                        [self.path_extra, "parameters/", "params_", self.params["save_file"], "-",
                         self.params["load_file"].lower(), ".npy"])

                    # Parameters to be preserved for params when charging
                    save, save_interval, num_tr, load_data, best_thr, eps_final, dropout, decay_lr, decay_lr_val, num_games, seed = \
                        self.params["save"], self.params["save_interval"], \
                        self.params["num_training"], self.params["load_data"], \
                        self.params["best_thr"], self.params["eps_final"], self.params["dropout"], self.params[
                            "dcy_lr"], \
                        self.params["dcy_lr_val"], self.params["num_games"], self.params["seed"]

                    # Load saved parameters
                    self.last_steps, self.accumTrainRewards, self.numeps, self.params, self.local_cnt, self.cnt, self.sub_dir = np.load(
                        load_path)  #
                    orig_num_training = self.params["num_training"]
                    # Load newest Parameters to params
                    orig_save_int = self.params["save_interval"]
                    self.params["save"], self.params["save_interval"], self.params["num_training"], \
                    self.params["load_data"], self.params["best_thr"], self.params["eps_final"], self.params["dropout"], \
                    self.params["decay_lr"], self.params["decay_lr_val"], self.params["num_games"], self.params[
                        "seed"] = save, save_interval, \
                                  num_tr, load_data, best_thr, eps_final, \
                                  dropout, decay_lr, decay_lr_val, num_games, seed

                    if self.sub_dir == "best":
                        print("Best Parameters Restored")
                    else:
                        print("Parameters Restored")

                    if self.params["load_data"]:  # Load data and starts with correct data
                        self.params["num_training"] += orig_num_training
                        if not self.params["load_file"].lower() == "best":
                            src = "".join(
                                [self.path_extra, "data/mem_rep_", self.params["save_file"], "/",
                                 str(int(self.sub_dir) - orig_save_int)])
                            rm_dir = "".join(
                                [self.path_extra, "data/mem_rep_", self.params["save_file"], "/", self.sub_dir])
                            del_dir(rm_dir)
                            copy_mem_rep(src=src,
                                         dst="".join([self.path_extra, "data/mem_rep_", self.params["save_file"], "/",
                                                      self.sub_dir]))
                            print("Data Restored")
                        else:
                            print("Best Data Restored")
                            print("Directory,", "".join(
                                [self.path_extra, "data/mem_rep_", self.params["save_file"], "/", self.sub_dir]))
                except Exception as e:
                    print(e)
                    print("Parameters don't exist or could not be properly loaded")
            except:
                print("Model don't exist or could not be properly loaded")

    def save_mod(self, best_mod=False):
        """
        Saving model and parameters
        Possibility of saving the best model
        """
        if (self.numeps % self.params["save_interval"] == 0 and self.params["save"]) or (
                best_mod and self.params["save"]):
            self.params["global_step"] = self.cnt
            save_files = [self.last_steps, self.accumTrainRewards, self.numeps, self.params, self.local_cnt, self.cnt]
            try:
                if best_mod:
                    self.saver.save(self.sess,
                                    "".join([self.path_extra, "model/", self.params["save_file"], "-", "best"]))
                    print("Best Model Saved")
                elif not self.sub_dir == "best":
                    self.saver.save(self.sess, "".join(
                        [self.path_extra, "model/", self.params["save_file"], "-", str(self.numeps)]))
                    print("Model Saved")
            except Exception as e:
                print("Model could not be saved")
                print("Error", e)
            try:
                if str(self.numeps) == self.sub_dir:  # Save memory replay and parameters
                    dst = "".join([self.path_extra, "data/mem_rep_", self.params["save_file"], "/",
                                   str(self.numeps + self.params["save_interval"])])
                    old_src = "".join([self.path_extra, "data/mem_rep_", self.params["save_file"], "/",
                                       str(self.numeps - self.params["save_interval"])])
                    del_dir(dst)
                    del_dir(old_src)
                    copy_mem_rep(
                        src="".join([self.path_extra, "data/mem_rep_", self.params["save_file"], "/", self.sub_dir]),
                        dst=dst)
                    self.sub_dir = str(self.numeps + self.params["save_interval"])
                    save_files.append(self.sub_dir)
                    print("Memory Replay Saved")
                    np.save("".join(
                        [self.path_extra, "parameters/", "params_", self.params["save_file"], "-", str(self.numeps)]),
                        save_files)
                    print("Pameters Saved")
                elif best_mod:  # Save memory replay of best model in directory "best" and parameters
                    if not self.sub_dir == "best":
                        dst = "".join([self.path_extra, "data/mem_rep_", self.params["save_file"], "/",
                                       "best"])
                        del_dir(dst)
                        copy_mem_rep(src="".join(
                            [self.path_extra, "data/mem_rep_", self.params["save_file"], "/", self.sub_dir]),
                            dst=dst)
                    save_files.append("best")
                    print("Best Memory Replay Saved")
                    np.save("".join([self.path_extra, "parameters/", "params_", self.params["save_file"], "-", "best"]),
                            save_files)
                    print("Best Pameters Saved")
            except Exception as e:
                print("Parameters could not be saved")
                print("Error", e)
