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

import gym
import gym_gfetch

class skills_extractor_Mj():
	def __init__(self, demo_path, env, eps_state=0.5, beta=1.25):

		self.env = env
		self.num_envs = env.num_envs
		self.device = env.device

		self.eps_state = eps_state ## threshold distance in goal space for skill construction
		self.beta = beta

		self.L_observations, self.L_sim_states = self.get_demo(demo_path)

		self.L_observations = self.L_observations[:150]
		self.L_sim_states = self.L_sim_states[:150]

		self.demo_length = len(self.L_observations)
		# print("len(L_obs) = ", len(self.L_observations))
		# print("len(L_sim) = ", len(self.L_sim_states))
		# print("demo_length = ", self.demo_length)
		# print("self.L_observations = ", self.L_observations)
		self.skills_sequence, self.demo_length = self.get_skills(self.L_observations, self.L_sim_states)

		# print("demo_length = ", self.demo_length)

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
		L_sim_states = []

		assert os.path.isfile(demo_path)

		with open(demo_path, "rb") as f:
			demo = pickle.load(f)
			# print("demo.keys() = ", demo.keys())
		for obs, sim_state in zip(demo["observations"], demo["sim_states"]):
			L_observations.append(obs)
			L_sim_states.append(sim_state)
		return L_observations, L_sim_states

	def get_skills(self, L_observations, L_sim_states):
		"""
		Clean the demonstration trajectory according to hyperparameter eps_dist.
		"""
		skills_sequence = []
		self.env.set_state([L_sim_states[0]], np.ones((1,)))

		# print("self.env.state = ", self.env.state)

		curr_state = self.env.state_vector()

		assert curr_state.all() == L_observations[0].all()

		curr_starting_state = (curr_state.copy(), L_sim_states[0])

		demo_length = 0

		i = 0
		while i < len(L_sim_states)-1:
			k = 0
			sum_dist = 0

			# cumulative distance
			while sum_dist <= self.eps_state and i + k < len(L_observations) - 1:
				self.env.set_state([L_sim_states[i+k]], np.ones((1,)))
				shifted_state = self.env.state_vector()
				# print("shifted_state = ", shifted_state)
				# print("curr_state = ", curr_state)
				sum_dist += self.env.goal_distance(self.env.project_to_goal_space(shifted_state), self.env.project_to_goal_space(curr_state))
				# print("sum_dist = ", sum_dist)
				curr_state = shifted_state.copy()
				k += 1
				demo_length += 1

			# skills_sequence.append((curr_starting_state, int(self.beta*k), shifted_state.copy()))
			skills_sequence.append((curr_starting_state, max(int(self.beta*k), 100), shifted_state.copy()))
			i = i + k

			print("k = ", k)

			# print("\ncurr_state[:10] = ", curr_state[:50])
			# print("L_observations[i][:10] = ", L_observations[i-1][:50])
			# print("test eq = ", curr_state[:100] == L_observations[i-1][:100])

			## check that curr_state corresponds to observation
			# assert (curr_state == L_observations[i-1]).all()
			# print("curr_state == ")
			# print("\ncurr_state[:10] = ", curr_state[:10])
			# print("L_observations[i][:10] = ", L_observations[i][:10])
			# print("L_sim_states[i][3][:10] = ", L_sim_states[i][3][:10])
			# assert (L_observations[i] == L_sim_states[i][3]).all()

			curr_starting_state = (curr_state, L_sim_states[i-1])

		# print("len(skills_sequence) = ", len(skills_sequence))
		return skills_sequence, demo_length


if (__name__=='__main__'):

	demo_path = "../demos/fetch_convert/1.demo"

	eval_env = gym.make("GFetchGoal-v0")

	## missing variables because we miss a wrapper here
	eval_env.num_envs = 1
	eval_env.device = "cpu"

	s_extractor = skills_extractor(demo_path, eval_env)

	print(s_extractor)
