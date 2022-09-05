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
	save_path = "/Users/chenu/Desktop/PhD/github/dcil_xpag/demos/toymaze/1.demo"

	L_observations = [np.array([0.5,0.5]), np.array([2.5, 1.]), np.array([1.5, 2.4])]
	L_full_states = [np.array([0.5,0.5]), np.array([2.5, 1.]), np.array([1.5, 2.4])]

	save_demo_dubins(save_path, L_observations, L_full_states)
