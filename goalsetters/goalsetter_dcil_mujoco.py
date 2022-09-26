# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

from abc import ABC, abstractmethod
from typing import Tuple

import xpag
from xpag.goalsetters import GoalSetter

import numpy as np

from operator import itemgetter

class DCILGoalSetterMj(GoalSetter, ABC):
	def __init__(self, do_overshoot = True):
		super().__init__("DCILGoalSetter")

		self.skills_sequence = []
		self.nb_skills = 0

		self.curr_indx = None
		self.reset_indx = None

		self.do_overshoot = do_overshoot

		self.weighted_sampling = True
		self.window_size = 10


	def step(
		self,
		env,
		observation,
		action,
		new_observation,
		reward,
		done,
		info,
		eval_mode=False,
	):

		## get next skill goal for bonus reward computation
		info["skill_indx"] = self.curr_indx.copy()
		info["next_skill_indx"] = self.next_indx.copy()
		info["reset_skill_indx"] = self.reset_indx.copy()

		new_obs = self.get_observation(env)

		info["next_skill_goal"] = self.skills_goals[info["next_skill_indx"].reshape(-1),0,:]

		# assert (new_obs["skill_indx"] == info["skill_indx"]).all()
		# assert (new_obs["next_skill_indx"] == info["next_skill_indx"]).all()

		self.last_info = info.copy()
		self.last_done = done

		# print("reward = ", reward)
		# print("info['is_success'] = ", info["is_success"])

		assert reward.max() <= 1
		assert reward.min() >= 0

		# assert reward.max() <= 0

		return new_obs, reward, done, info

	def write_config(self, output_file: str):
		pass

	def save(self, directory: str):
		pass

	def load(self, directory: str):
		pass

	def set_skills_sequence(self, sseq, env, n_skills=None):
		"""
		sseq = [skill_1, skill_2, ...]
		skill_i = ((starting_observation, starting_sim_state), skill_length, skill_goal)
		"""
		if n_skills is None:
			self.skills_sequence = sseq
		else:
			self.skills_sequence = sseq[:n_skills]

		self.nb_skills = len(self.skills_sequence)
		self.curr_indx = np.zeros((env.num_envs,1)).astype(np.intc)

		## monitor skill results
		self.L_skills_results = [[] for _ in self.skills_sequence]

		self.skills_observations = np.zeros((self.nb_skills, env.num_envs, env.get_obs_dim()))
		self.skills_sim_states = []
		self.skills_max_episode_steps = np.zeros((self.nb_skills, env.num_envs, 1))
		self.skills_goals = np.zeros((self.nb_skills, env.num_envs, env.get_goal_dim()))

		for i,skill in enumerate(self.skills_sequence):
			skill_starting_state, skill_length, skill_goal_state = skill
			observation, sim_state = skill_starting_state

			self.skills_observations[i,:,:] = np.tile(np.array(observation).flatten(), (env.num_envs, 1))

			# list of n_skills x [num_envs x MjSimState]
			# Note: cannot use array of state directly as object are a bit more complicated (qpos , qvel, time, etc.)
			self.skills_sim_states.append([sim_state for i in range(env.num_envs)])

			self.skills_max_episode_steps[i,:,:] = np.tile(np.array(skill_length).flatten(), (env.num_envs, 1))
			self.skills_goals[i,:,:] = np.tile(env.project_to_goal_space(skill_goal_state).flatten(), (env.num_envs, 1))

		self.convert_table = np.eye(self.nb_skills)

		return 0

	def get_skills_success_rates(self):

		L_rates = []
		for i in range(0, len(self.L_skills_results)):
			L_rates.append(self.get_skill_success_rate(i))

		return L_rates

	def get_skill_success_rate(self, skill_indx):

		nb_skills_success = self.L_skills_results[skill_indx].count(1)
		s_r = float(nb_skills_success/len(self.L_skills_results[skill_indx]))

		## keep a small probability for successful skills to be selected in order
		## to avoid catastrophic forgetting
		if s_r <= 0.1:
			s_r = 10
		else:
			s_r = 1./s_r

		return s_r

	def sample_skill_indx(self, n_envs):
		"""
		Sample a skill indx (weighted sampling of skill according to skill success rate or uniform sampling)
		"""

		weights_available = True
		for i in range(0,len(self.L_skills_results)):
			if len(self.L_skills_results[i]) == 0:
				weights_available = False

		# # print("self.L_skills_results = ", self.L_skills_results)
		# #print("weights available = ", weights_available)
		#
		# ## fitness based selection
		if self.weighted_sampling and weights_available:

			L_rates = self.get_skills_success_rates()

			## weighted sampling
			total_rate = sum(L_rates)

			L_new_skill_indx = []
			for i_env in range(n_envs):
				pick = np.random.uniform(0, total_rate)

				current = 0
				indx_thresh = 0
				for i in range(0,len(L_rates)):
					s_r = L_rates[i]
					current += s_r
					if current > pick:
						break
					indx_thresh += 1

				L_new_skill_indx.append([indx_thresh])

			## TODO: switch for tensor version
			new_skill_indx = np.array(L_new_skill_indx)

			assert new_skill_indx.shape == (n_envs, 1)

		## uniform sampling
		else:
			new_skill_indx = np.random.randint(0, self.nb_skills, (n_envs,)).reshape(n_envs,1)

		# print("new_skill_indx = ", new_skill_indx)

		return new_skill_indx.astype(np.intc)

	def shift_skill(self, env):
		is_done = self.last_done
		next_skill_indices, next_skill_avail, next_next_indx, next_next_skill_avail = self.next_skill_indx()

		## if overshoot possible, choose next skill indx, otherwise, sample new skill indx
		self.curr_indx = np.where(next_skill_avail == 1, next_skill_indices, self.curr_indx)
		next_indx = self.curr_indx + 1
		self.next_indx = np.where(next_indx < self.nb_skills, next_indx, self.curr_indx)

		## recover skill
		reset_max_episode_steps = self.skills_max_episode_steps[self.curr_indx.reshape(-1),0,:]
		reset_goals = self.skills_goals[self.curr_indx.reshape(-1),0,:]

		## set skill
		do_reset_max_episode_steps = is_done.copy()
		env.set_max_episode_steps(reset_max_episode_steps, do_reset_max_episode_steps)
		do_reset_goal = is_done.copy()
		env.set_goal(reset_goals, do_reset_goal)

		return self.get_observation(env), next_skill_avail

	def next_skill_indx(self, eval=True):
		curr_indx = self.curr_indx.copy()

		next_indx = curr_indx + 1
		next_skill_avail = (next_indx < self.nb_skills).astype(np.intc)

		next_next_indx = curr_indx + 2
		next_next_skill_avail = (next_next_indx < self.nb_skills).astype(np.intc)

		return next_indx, next_skill_avail, next_next_indx, next_next_skill_avail

	def _select_skill_indx(self, is_success, n_envs):
		"""
		Select skills (starting state, budget and goal)
		"""
		sampled_skill_indices = self.sample_skill_indx(n_envs) ## return tensor of new indices
		next_skill_indices, next_skill_avail, _, _ = self.next_skill_indx() ## return tensor of next skills indices

		overshoot_possible = np.logical_and(is_success, next_skill_avail).astype(np.intc) * int(self.do_overshoot)

		## if overshoot possible, choose next skill indx, otherwise, sample new skill indx
		goal_skill_indices = np.where(overshoot_possible == 1, next_skill_indices, sampled_skill_indices)

		return goal_skill_indices, overshoot_possible

	def reset(self, env, observation, eval_mode=False):
		## reset to first skill
		self.curr_indx = np.zeros((env.num_envs,1)).astype(np.intc)
		self.next_indx = np.ones((env.num_envs,1)).astype(np.intc)

		self.reset_indx = self.curr_indx.copy()

		## recover skill
		reset_observations = self.skills_observations[self.curr_indx.reshape(-1),0,:]
		reset_sim_states = [item for item in itemgetter(*self.curr_indx.reshape(-1))(self.skills_sim_states)]
		# reset_sim_states = self.skills_sim_states[self.curr_indx.reshape(-1)][0]
		reset_max_episode_steps = self.skills_max_episode_steps[self.curr_indx.reshape(-1),0,:]
		reset_goals = self.skills_goals[self.curr_indx.reshape(-1),0,:]

		## set skill
		do_reset_state = np.ones((env.num_envs,1))
		env.set_state(reset_sim_states, do_reset_state)
		do_reset_max_episode_steps = np.ones((env.num_envs,1))
		env.set_max_episode_steps(reset_max_episode_steps, do_reset_max_episode_steps)
		do_reset_goal = np.ones((env.num_envs,1))
		env.set_goal(reset_goals, do_reset_goal)

		return self.get_observation(env)

	def add_success_and_failures(self, is_done, is_success):

		## get indx of success and fail in batch
		is_done_not_success = np.logical_and(is_done.flatten(), np.logical_not(is_success.flatten())).astype(np.intc)
		success_indx = np.where(is_success.flatten() == 1)[0]
		fail_indx = np.where(is_done_not_success == 1)[0]

		# print("is_done_not_success = ", is_done_not_success)
		# print("success_indx = ", success_indx)
		# print("fail_indx = ", fail_indx)

		for s_indx in list(success_indx):
			# print("s_indx = ", s_indx)
			self.L_skills_results[self.curr_indx.flatten()[s_indx]].append(1)
			if len(self.L_skills_results[self.curr_indx.flatten()[s_indx]]) > self.window_size:
				self.L_skills_results[self.curr_indx.flatten()[s_indx]].pop(0)

		for f_indx in list(fail_indx):
			# print("f_indx = ", f_indx)
			self.L_skills_results[self.curr_indx.flatten()[f_indx]].append(0)
			if len(self.L_skills_results[self.curr_indx.flatten()[f_indx]]) > self.window_size:
				self.L_skills_results[self.curr_indx.flatten()[f_indx]].pop(0)

		return


	def reset_done(self, env, observation, eval_mode=False):

		is_done = self.last_done
		is_success = self.last_info["is_success"] ## array of booleans

		## update skills scores
		# print("\n skills_results before = ", self.L_skills_results)
		self.add_success_and_failures(is_done, is_success)
		# print("skills_results after = ", self.L_skills_results)

		## sample new skill
		select_skill_indx, overshoot_possible = self._select_skill_indx(is_success, env.num_envs)

		r = np.random.rand(is_success.shape[0], is_success.shape[1])
		start_skill_indx = select_skill_indx.copy()
		skipping_indx = np.where(r>1., select_skill_indx+1, select_skill_indx) ## skipping for 10% of rollouts
		select_skill_indx = np.where(skipping_indx < self.nb_skills, skipping_indx, select_skill_indx)

		## TODO: fix indx management for multiple environments
		# ## if overshoot possible (done & success & curr_indx + 1 < nb_skills) -> change for next indx if possible
		self.curr_indx = select_skill_indx
		next_skill_indx = self.curr_indx + 1
		self.next_indx = np.where(next_skill_indx < self.nb_skills, next_skill_indx, self.curr_indx)

		## update reset indx to curr indx if no overshoot
		self.reset_indx = np.where(np.logical_and(is_done, np.logical_not(overshoot_possible)).astype(np.intc)==1, self.curr_indx, self.reset_indx)

		## recover skill
		reset_observations = self.skills_observations[start_skill_indx.reshape(-1),0,:]
		reset_sim_states = [item for item in itemgetter(*start_skill_indx.reshape(-1))(self.skills_sim_states)]
		reset_max_episode_steps = self.skills_max_episode_steps[self.curr_indx.reshape(-1),0,:]
		reset_goals = self.skills_goals[self.curr_indx.reshape(-1),0,:]

		## set skill
		do_reset_state = np.logical_and(is_done, np.logical_not(overshoot_possible)).astype(np.intc)
		env.set_state(reset_sim_states, do_reset_state)
		do_reset_max_episode_steps = is_done.copy()
		env.set_max_episode_steps(reset_max_episode_steps, do_reset_max_episode_steps)
		do_reset_goal = is_done.copy()
		env.set_goal(reset_goals, do_reset_goal)

		return self.get_observation(env)

	def get_observation(self, env):
		obs = env.get_observation()
		next_skill_indx = self.next_indx.copy()
		obs["next_skill_goal"] = self.skills_goals[next_skill_indx.reshape(-1),0,:]

		return obs

# if __name__ == "__main__":

	# demo_path = "../demos/fetch_convert/1.demo"
	# eval_env = gym.make("GFetchGoal-v0")
	# ## missing variables because we miss a wrapper here
	# eval_env.num_envs = 1
	# eval_env.device = "cpu"
	#
	# s_extractor = skills_extractor(demo_path, eval_env)
	#
	# goalsetter = DCILGoalSetterMj()
