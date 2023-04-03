
from torch.autograd import Variable
import random, pickle
import numpy as np
import torch
import os
import gym


class Tracker:
    def __init__(self, parameters, vars_string, project_string):
        self.vars_string = vars_string; self.project_string = project_string
        self.foldername = parameters.save_foldername
        self.period = parameters.save_csv_freq
        # [Id of var tracked][fitnesses, avg_fitness, csv_avg_fitnesses, csv_test_fitnesses]
        self.all_tracker = [[[],0.0,[],[]] for _ in vars_string]
        self.counter = 0
        self.conv_size = 10
        if not os.path.exists(self.foldername):
            os.makedirs(self.foldername)

    def update(self, updates, generation):
        self.counter += 1
        for update, var in zip(updates, self.all_tracker):
            if update == None: continue
            var[0].append(update)

        # Constrain size of convolution
        for var in self.all_tracker:
            if len(var[0]) > self.conv_size: var[0].pop(0)

        # Update new average
        for var in self.all_tracker:
            if len(var[0]) == 0: continue
            var[1] = sum(var[0])/float(len(var[0]))

        if self.counter % self.period == 0:  # Save to csv file
            for i, var in enumerate(self.all_tracker):
                if len(var[0]) == 0: continue
                var[2].append(np.array([generation, var[1]]))
                var[3].append(np.array([generation, var[0][-1]]))
                avg_filename = os.path.join(self.foldername, self.vars_string[i] + "_avg" + self.project_string)
                test_filename = os.path.join(self.foldername, self.vars_string[i] + "_test" + self.project_string)
                try:
                    np.savetxt(avg_filename, np.array(var[2]), fmt='%.3f', delimiter=',')
                    np.savetxt(test_filename, np.array(var[3]), fmt='%.3f', delimiter=',')
                except:
                    # Common error showing up in the cluster for unknown reasons
                    print('Failed to save progress')

class NormalizedActions(gym.ActionWrapper):

    def action(self, action):
        action = (action + 1) / 2  # [-1, 1] => [0, 1]
        action *= (self.action_space.high - self.action_space.low)
        action += self.action_space.low
        return action

    def _reverse_action(self, action):
        action -= self.action_space.low
        action /= (self.action_space.high - self.action_space.low)
        action = action * 2 - 1
        return action


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    #v = 1. / np.sqrt(fanin)
    v = 0.008
    return torch.Tensor(size).uniform_(-v, v)

def to_numpy(var):
    return var.data.numpy()

def to_tensor(ndarray, volatile=False, requires_grad=False):
    return Variable(torch.from_numpy(ndarray).float(), volatile=volatile, requires_grad=requires_grad)

def pickle_obj(filename, object):
    handle = open(filename, "wb")
    pickle.dump(object, handle)

def unpickle_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def odict_to_numpy(odict):
    l = list(odict.values())
    state = l[0]
    for i in range(1, len(l)):
        if isinstance(l[i], np.ndarray):
            state = np.concatenate((state, l[i]))
        else: #Floats
            state = np.concatenate((state, np.array([l[i]])))
    return state

def min_max_normalize(x):
    min_x = np.min(x)
    max_x = np.max(x)
    return (x - min_x) / (max_x - min_x)

def is_lnorm_key(key):
    return key.startswith('lnorm')

# Print text with different colors
def prRed(prt):
    print("\033[91m{}\033[00m" .format(prt))


def prGreen(prt):
    print("\033[92m{}\033[00m" .format(prt))


def prYellow(prt):
    print("\033[93m{}\033[00m" .format(prt))


def prLightPurple(prt):
    print("\033[94m{}\033[00m" .format(prt))


def prPurple(prt):
    print("\033[95m{}\033[00m" .format(prt))


def prCyan(prt):
    print("\033[96m{}\033[00m" .format(prt))


def prLightGray(prt):
    print("\033[97m{}\033[00m" .format(prt))


def prBlack(prt):
    print("\033[98m{}\033[00m" .format(prt))

