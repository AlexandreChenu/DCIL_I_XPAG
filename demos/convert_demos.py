import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import math
import random
import copy
from collections import OrderedDict
import torch
import pickle


def get_demo_dubins( demo_path):
    """
    Extract demo from pickled file
    """
    L_observations = []
    L_full_states = []

    assert os.path.isfile(demo_path)

    with open(demo_path, "rb") as f:
        demo = pickle.load(f)
        print("demo.keys() = ", demo.keys())
    print("len(demo['obs']) = ", len(demo["obs"]))
    
    for obs in zip(demo["obs"]):
        print("obs = ", obs)
        L_observations.append(np.array(obs[0]))
        L_full_states.append(np.array(obs[0]))

    return L_observations, L_full_states

def save_demo_dubins(save_path, L_observations, L_full_states):
    """
    save converted demo
    """
    demo = {}
    demo["observations"] = L_observations
    demo["full_states"] = L_full_states

    print("length observations = ", len(demo["observations"]))
    print("length full states = ", len(demo["full_states"]))

    print("observations[0] = ", demo["observations"][0])
    print("full_states[0] = ", demo["full_states"][0])

    with open(save_path, 'wb') as handle:
        pickle.dump(demo, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return

if __name__ == '__main__':
    demo_path = "/Users/chenu/Desktop/PhD/github/dcil_xpag/demos/dubins/1.demo"
    save_path = "/Users/chenu/Desktop/PhD/github/dcil_xpag/demos/dubins_convert/1.demo"

    L_observations, L_full_states = get_demo_dubins(demo_path)

    save_demo_dubins(save_path, L_observations, L_full_states)
