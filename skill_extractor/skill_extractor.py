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

import matplotlib.pyplot as plt

class skills_extractor():
	def __init__(self, demo_path, env, eps_state=1., beta=1.25):

		self.env = env
		self.num_envs = env.num_envs
		self.device = env.device

		self.eps_state = eps_state ## threshold distance in goal space for skill construction
		self.beta = beta

		self.L_observations, self.L_sim_states = self.get_demo(demo_path)
		print("L_observations = ", self.L_observations)
		self.skills_sequence = self.get_skills(self.L_observations, self.L_sim_states)

		## visual test
		# self.test_visu()

	def test_visu(self):
		fig, ax = plt.subplots()
		self.env.plot(ax)
		X = [obs[0] for obs in self.L_observations]
		Y = [obs[1] for obs in self.L_observations]
		ax.scatter(X, Y, c = "blue", marker=".")
		X_start = []
		Y_start = []
		for skill in self.skills_sequence:
			# print("skill = ", skill)
			state, _, _ = skill
			# print("state = ", state[0])
			X_start.append(state[0][0][0])
			Y_start.append(state[0][0][1])
		ax.scatter(X_start, Y_start, c="red", marker="o")
		X_goal = []
		Y_goal = []
		for skill in self.skills_sequence:
			# print("skill = ", skill)
			_, _, goal = skill
			# print("state = ", state[0])
			X_goal.append(goal[0][0])
			Y_goal.append(goal[0][1])
		ax.scatter(X_goal, Y_goal, c="green", marker="o")
		plt.show()
		return

	def get_demo(self, demo_path, verbose=0):
		"""
		Extract demo from pickled file
		"""
		L_observations = []
		L_full_states = []

		assert os.path.isfile(demo_path)

		with open(demo_path, "rb") as f:
			demo = pickle.load(f)
			# print("demo.keys() = ", demo.keys())
		for obs, full_state in zip(demo["observations"], demo["full_states"]):
			L_observations.append(obs)
			L_full_states.append(full_state)
		return L_observations, L_full_states

	def get_skills(self, L_observations, L_sim_states):
		"""
		Clean the demonstration trajectory according to hyperparameter eps_dist.
		"""
		skills_sequence = []
		self.env.set_state(L_observations[0], np.ones((1,)))

		# print("self.env.state = ", self.env.state)

		curr_state = self.env.get_state()
		curr_starting_state = (curr_state.copy(), curr_state.copy())

		i = 0
		while i < len(L_sim_states):
			k = 0
			sum_dist = 0

			# cumulative distance
			while sum_dist <= self.eps_state and i + k < len(L_observations) :
				# print("sum_dist = ", sum_dist)
				# print("curr_state = ", curr_state)
				# print("k = ", k)
				self.env.set_state(L_observations[i+k], np.ones((1,)))
				shifted_state = self.env.get_state()
				# print("shifted_state = ", shifted_state)
				# print("curr_state = ", curr_state)
				sum_dist += self.env.goal_distance(self.env.project_to_goal_space(shifted_state), self.env.project_to_goal_space(curr_state))
				# print("sum_dist = ", sum_dist)
				curr_state = shifted_state.copy()
				k += 1

			# skills_sequence.append((curr_starting_state, int(self.beta*k), shifted_state.copy()))
			skills_sequence.append((curr_starting_state, 25, shifted_state.copy()))
			i = i + k
			# print("i = ", i)
			curr_starting_state = (curr_state, curr_state)

		# print("len(skills_sequence) = ", len(skills_sequence))
		return skills_sequence
