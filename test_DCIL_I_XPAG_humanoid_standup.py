#!/usr/bin python -w

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

import time
import jax

print(jax.lib.xla_bridge.get_backend().platform)

#jax.config.update('jax_platform_name', "cpu")

from datetime import datetime
import argparse

import flax

import xpag
# from xpag.wrappers import gym_vec_env

from xpag.buffers import DefaultEpisodicBuffer
# from xpag.samplers import DefaultEpisodicSampler, HER
from xpag.tools import mujoco_notebook_replay

from xpag.tools.eval import single_rollout_eval
from xpag.tools.utils import hstack
from xpag.tools.logging import eval_log_reset
from xpag.tools.timing import timing_reset
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import numpy as np
import copy

import gym_ghumanoid_standup

## DCIL versions
from wrappers.gym_vec_env_mujoco import gym_vec_env
from skill_extractor import skills_extractor_Mj
from samplers import HER_DCIL
from goalsetters import DCILGoalSetterMj
from agents import SAC_DCIL

import cv2
import pickle

import pdb

def visu_success_zones(eval_env, skill_sequence, ax):
	"""
	Visualize success zones as sphere of radius eps_success around skill-goals
	"""

	for skill in skill_sequence:
		starting_state, _, _ = skill
		obs, full_state = starting_state
		goal = eval_env.project_to_goal_space(obs)

		u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]

		x = goal[0] + 0.05*np.cos(u)*np.sin(v)
		y = goal[1] + 0.05*np.sin(u)*np.sin(v)
		z = goal[2] + 0.05*np.cos(v)
		ax.plot_wireframe(x, y, z, color="blue", alpha = 0.1)

	return

def plot_traj(eval_env, s_trajs, f_trajs, traj_eval, skill_sequence, save_dir, it=0):
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')

	for traj in s_trajs:
		# print("traj = ", traj)
		for i in range(traj[0].shape[0]):
			X = [eval_env.project_to_goal_space(state[i])[0] for state in traj]
			Y = [eval_env.project_to_goal_space(state[i])[1] for state in traj]
			Z = [eval_env.project_to_goal_space(state[i])[2] for state in traj]
			ax.plot(X,Y,Z, c="m", alpha = 0.4)

	for traj in f_trajs:
		# print("traj = ", traj)
		for i in range(traj[0].shape[0]):
			X = [eval_env.project_to_goal_space(state[i])[0] for state in traj]
			Y = [eval_env.project_to_goal_space(state[i])[1] for state in traj]
			Z = [eval_env.project_to_goal_space(state[i])[2] for state in traj]
			ax.plot(X,Y,Z, c="plum", alpha = 0.4)

	X_eval = [eval_env.project_to_goal_space(state[0])[0] for state in traj_eval]
	Y_eval = [eval_env.project_to_goal_space(state[0])[1] for state in traj_eval]
	Z_eval = [eval_env.project_to_goal_space(state[0])[2] for state in traj_eval]
	ax.plot(X_eval, Y_eval, Z_eval, c = "blue")

	visu_success_zones(eval_env, skill_sequence, ax)

	for _azim in range(45, 360, 90):
		ax.view_init(azim=_azim)
		plt.savefig(save_dir + "/trajs_azim_" + str(_azim) + "_it_" + str(it) + ".png")
	# plt.savefig(save_dir + "/trajs_it_"+str(it)+".png")
	plt.close(fig)
	return

import torch
@torch.no_grad()


def visu_value(env, eval_env, agent, skill_sequence):

		values = []

		convert_table = np.eye(len(skill_sequence))

		for skill_indx in range(0,len(skill_sequence)-1):

			obs = eval_env.reset()
			skill = skill_sequence[skill_indx]
			next_skill = skill_sequence[skill_indx+1]

			starting_state, _, desired_goal_state = skill
			desired_goal = eval_env.project_to_goal_space(desired_goal_state)
			_, _, next_desired_goal_state = next_skill
			next_desired_goal = eval_env.project_to_goal_space(next_desired_goal_state)

			observation, sim_state = starting_state

			## convert skill indx to a vectorized binary (TODO)
			oh_skill_indx = convert_table[skill_indx]

			eval_env.set_state([sim_state], np.ones((1,)))

			obs = eval_env.get_observation()

			if hasattr(env, "obs_rms"):
				action = agent.select_action(np.hstack((env._normalize_shape(obs["observation"],env.obs_rms["observation"]),
													env._normalize_shape(desired_goal.reshape(1,3),env.obs_rms["achieved_goal"]))),
					deterministic=True,
				)
				value = agent.value(np.hstack((env._normalize_shape(obs["observation"],env.obs_rms["observation"]),
										   env._normalize_shape(desired_goal.reshape(1,3),env.obs_rms["achieved_goal"]))), action)
			else:
				# print(obs["observation"].shape)
				# print(desired_goal.shape)
				# print(next_desired_goal.shape)
				action = agent.select_action(np.hstack((obs["observation"], desired_goal.reshape(1,3))),
					deterministic=True,
				)
				value = agent.value(np.hstack((obs["observation"], desired_goal.reshape(1,3))), action)
			values.append(value[0])

		return values

def save_frames_as_video(frames, path, iteration):

	video_name = path + '/video' + str(iteration) + '.mp4'
	height, width, layers = frames[0].shape
	#resize
	percent = 100
	width = int(frames[0].shape[1] * percent / 100)
	height = int(frames[0].shape[0] * percent / 100)

	video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))

	for frame in frames:
		width = int(frame.shape[1] * percent / 100)
		height = int(frame.shape[0] * percent / 100)
		dim = (width, height)
		resize_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
		cvt_frame = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2RGB)
		video.write(cvt_frame)

	video.release()
	cv2.destroyAllWindows()

	return

def save_sim_traj(sim_traj, path, iteration):

	with open(path + "/sim_traj_" + str(iteration) + ".pickle", 'wb') as handle:
		pickle.dump(sim_traj, handle, protocol=pickle.HIGHEST_PROTOCOL)

	return

def eval_traj(env, eval_env, agent, demo_length, goalsetter, save_video=False, save_sim_traj=False):
	traj = []
	traj_length = 0
	observation = goalsetter.reset(eval_env, eval_env.reset())
	eval_done = False

	frames = []
	sim_states = []

	sum_env_reward = 0

	next_skill_avail = True

	# while goalsetter.curr_indx[0] <= goalsetter.nb_skills and not eval_done:
	while traj_length < demo_length and not eval_done:
		# skill_success = False
		# print("curr_indx = ", goalsetter.curr_indx)
		max_steps = eval_env.get_max_episode_steps()
		# print("max_steps = ", max_steps)
		for i_step in range(0,int(max_steps[0])):
			traj_length += 1
			#print("eval_env.skill_manager.indx_goal = ", eval_env.skill_manager.indx_goal)
			traj.append(observation["observation"].copy())
			if hasattr(env, "obs_rms"):
				action = agent.select_action(np.hstack((env._normalize_shape(observation["observation"],env.obs_rms["observation"]),
													env._normalize_shape(observation["desired_goal"],env.obs_rms["achieved_goal"]))),
					deterministic=True,
				)
			else:
				action = agent.select_action(np.hstack((observation["observation"], observation["desired_goal"])),
				deterministic=True,
				)

			# print("\neval_env.envs[0] = ", id(eval_env.envs[0]))
			# print("eval_env.envs = ", id(eval_env.envs))
			# print("eval_env = ", id(eval_env))
			# print("eval_env.env = ", id(eval_env.env))
			# print("eval_env.env.envs[0] = ", id(eval_env.env.envs[0]))
			# print("eval_env.env.envs = ", id(eval_env.env.envs))

			if save_video:
				frame = eval_env.envs[0].sim.render(width=1080, height=1080, mode="offscreen")
				# print("frame = ", frame)
				frames.append(frame)

			if save_sim_traj:
				sim_state = eval_env.envs[0].env.get_inner_state()
				# print("sim_state = ", sim_state)
				sim_states.append(sim_state)

			# print("action = ", action)
			observation, _, done, info = goalsetter.step(
				eval_env, observation, action, *eval_env.step(action)
			)

			sum_env_reward += info["reward_from_env"][0]


			# print("observation eval = ", observation["observation"][0][:15])
			# print("observation.shape = ", observation["observation"].shape)
			# print("observation = ", eval_env.project_to_goal_space(observation["observation"].reshape(268,)))
			# print("done = ", done)
			if done.max():
				observation, next_skill_avail = goalsetter.shift_skill(eval_env)
				break
			if traj_length >= demo_length:
				break

		if not next_skill_avail:
			eval_done = True

	return traj, frames, sim_states, sum_env_reward


if (__name__=='__main__'):

	parser = argparse.ArgumentParser(description='Argument for DCIL')
	parser.add_argument('--demo_path', help='path to demonstration file')
	parser.add_argument('--save_path', help='path to save directory')
	parser.add_argument('--eps_state', help='distance between 2 consecutive goal')
	parser.add_argument('--value_clipping', help='add value clipping in critic update?')

	parsed_args = parser.parse_args()

	env_args = {}
	env_args["demo_path"] = str(parsed_args.demo_path)

	num_envs = 1  # the number of rollouts in parallel during training
	env, eval_env, env_info = gym_vec_env('GHumanoidGoal_standup-v0', num_envs)
	print("env = ", env)
	num_skills = None


	s_extractor = skills_extractor_Mj(parsed_args.demo_path, eval_env, eps_state=float(parsed_args.eps_state))
	print("nb_skills (remember to adjust value clipping in sac_from_jaxrl)= ", len(s_extractor.skills_sequence))

	if num_skills == None:
		num_skills = len(s_extractor.skills_sequence)

	goalsetter = DCILGoalSetterMj()
	goalsetter.set_skills_sequence(s_extractor.skills_sequence, env, n_skills=num_skills)
	eval_goalsetter = DCILGoalSetterMj()
	eval_goalsetter.set_skills_sequence(s_extractor.skills_sequence, eval_env, n_skills=num_skills)

	# print(goalsetter.skills_observations)
	# print(goalsetter.skills_full_states)
	# print(goalsetter.skills_max_episode_steps)
	# print("goalsetter.skills_sequence = ", goalsetter.skills_sequence)

	batch_size = 64
	gd_steps_per_step = 1.5
	start_training_after_x_steps = env_info['max_episode_steps'] * 50
	max_steps = 1_000_000
	evaluate_every_x_steps = 2_000
	save_agent_every_x_steps = 50_000

	## create log dir
	now = datetime.now()
	dt_string = 'DCIL_I_' + str(parsed_args.eps_state) + "_" + str(bool(parsed_args.value_clipping)) + '_%s_%s' % (datetime.now().strftime('%Y%m%d'), str(os.getpid()))
	# save_dir = os.path.join('/gpfswork/rech/kcr/ubj56je', 'results', 'xpag', 'DCIL_XPAG_dubins', dt_string)
	# save_dir = os.path.join(os.path.expanduser('~'), 'results', 'xpag', 'DCIL_XPAG_dubins', dt_string)
	save_dir = str(parsed_args.save_path) + dt_string
	os.mkdir(save_dir)
	## log file for success ratio
	f_ratio = open(save_dir + "/ratio.txt", "w")
	f_critic_loss = open(save_dir + "/critic_loss.txt", "w")
	f_values = open(save_dir + "/value_start_states.txt", "w")
	f_total_eval_reward = open(save_dir + "/total_eval_reward.txt", "w")

	save_episode = True
	plot_projection = None
	do_save_video = False
	do_save_sim_traj = True

	params = {
		"actor_lr": 0.0003,
		"backup_entropy": False,
		"value_clipping": bool(parsed_args.value_clipping),
		"critic_lr": 0.0003,
		"discount": 0.98,
		"hidden_dims": (512, 512, 512),
		# "hidden_dims": (400,300),
		"init_temperature": 0.001,
		"target_entropy": None,
		"target_update_period": 1,
		"tau": 0.005,
		"temp_lr": 0.0003,
	}

	with open(save_dir + "/sac_params.txt", "w") as f:
		print(params, file=f)

	agent = SAC_DCIL(
		env_info['observation_dim'] if not env_info['is_goalenv']
		else env_info['observation_dim'] + env_info['desired_goal_dim'] ,
		env_info['action_dim'],
		params=params
	)
	sampler = DefaultEpisodicSampler() if not env_info['is_goalenv'] else HER_DCIL(env.envs[0].compute_reward, env)
	buffer_ = DefaultEpisodicBuffer(
		max_episode_steps=env_info['max_episode_steps'],
		buffer_size=1_000_000,
		sampler=sampler
	)

	eval_log_reset()
	timing_reset()
	observation = goalsetter.reset(env, env.reset())
	print("observation = ", observation)
	s_trajs = []
	f_trajs = []
	traj = []
	info_train = None
	num_success = 0
	num_rollouts = 0
	num_success_skill = np.zeros((goalsetter.nb_skills,goalsetter.nb_skills)).astype(np.intc)
	num_rollouts_skill = np.zeros((goalsetter.nb_skills,goalsetter.nb_skills)).astype(np.intc)


	for i in range(max_steps // env_info["num_envs"]):
		# print("learn: ", eval_env.project_to_goal_space(observation["observation"][0]))
		traj.append(observation["observation"].copy())
		# print("\n")

		if not i % max(evaluate_every_x_steps // env_info["num_envs"], 1):
			print("------------------------------------------------------------------------------------------------------------")
			print("| training steps nb ", i)
			# t1_logs = time.time()
			print("|")

			if hasattr(env, "obs_rms"):
				print("| do update ? ", env.do_update)
				print("| RMS = ", env.obs_rms["observation"].mean[0][:10])
			# single_rollout_eval(
			# 	i * env_info["num_envs"],
			# 	eval_env,
			# 	env_info,
			# 	agent,
			# 	save_dir=save_dir,
			# 	plot_projection=plot_projection,
			# 	save_episode=save_episode,
			# )
			traj_eval, frames, sim_traj, total_env_reward = eval_traj(env, eval_env, agent, s_extractor.demo_length, eval_goalsetter, save_video=do_save_video, save_sim_traj=do_save_sim_traj)
			if do_save_video:
				save_frames_as_video(frames, save_dir, i)
			if do_save_sim_traj:
				save_sim_traj(sim_traj, save_dir, i)

			print("| cumulative env reward = ", total_env_reward)

			f_total_eval_reward.write(str(total_env_reward) + "\n")

			# print("traj_eval = ", traj_eval)
			plot_traj(eval_env, s_trajs, f_trajs, traj_eval, eval_goalsetter.skills_sequence, save_dir, it=i)
			values = visu_value(env, eval_env, agent, eval_goalsetter.skills_sequence)
			print("| values = ", values)
			for value in values:
				f_values.write(str(value) + " ")
			f_values.write("\n")

			# t2_logs = time.time()
			# print("logs time = ", t2_logs - t1_logs)

			# if i > 2000:
				# visu_transitions(eval_env, transitions, it = i)
				# print("info_train = ", info_train)
			s_trajs = []
			f_trajs = []
			traj = []
			# if info_train is not None:
			# 	print("rewards = ", max(info_train["rewards"]))

			if num_rollouts > 0:
				print("| success ratio (successful skill-rollouts / total rollouts) : ", float(num_success/num_rollouts))
				print("| skills success : ", [np.array(result).mean() for result in goalsetter.L_skills_results])
				print("| overshoot success : ")
				# print("| num_success_skill = ", np.array(num_success_skill))
				# print("| num_rollouts_skill = ", np.array(num_rollouts_skill))
				np.savetxt(save_dir + "/success_skill_" + str(i) + ".txt", num_success_skill)
				np.savetxt(save_dir + "/rollout_skill_" + str(i) + ".txt", num_rollouts_skill)
				f_ratio.write(str(float(num_success/num_rollouts)) + "\n")
				num_success = 0
				num_rollouts = 0
				num_success_skill = np.zeros((goalsetter.nb_skills,goalsetter.nb_skills)).astype(np.intc)
				num_rollouts_skill = np.zeros((goalsetter.nb_skills,goalsetter.nb_skills)).astype(np.intc)

			print("------------------------------------------------------------------------------------------------------------")

		if not i % max(save_agent_every_x_steps // env_info["num_envs"], 1):
			if save_dir is not None:
				agent.save(os.path.join(save_dir, "agent"))

		if i * env_info["num_envs"] < start_training_after_x_steps:
			action = env_info["action_space"].sample()
		else:
			env.do_update = False
			t1_a_select = time.time()
			if hasattr(env, "obs_rms"):
				action = agent.select_action(
					observation
					if not env_info["is_goalenv"]
					else np.hstack((env._normalize(observation["observation"], env.obs_rms["observation"]),
									env._normalize(observation["desired_goal"], env.obs_rms["achieved_goal"]))),
					deterministic=False,
				)
			else:
				action = agent.select_action(
					observation
					if not env_info["is_goalenv"]
					else np.hstack((observation["observation"], observation["desired_goal"])),
					deterministic=False,
				)
			t2_a_select = time.time()
			# print("action selection time = ", t2_a_select - t1_a_select)

			t1_train = time.time()
			for _ in range(max(round(gd_steps_per_step * env_info["num_envs"]), 1)):
				transitions = buffer_.sample(batch_size)
				info_train = agent.train_on_batch(transitions)
			t2_train = time.time()
			# print("training time = ", t2_train - t1_train)

			if i % 100 == 0:
				f_critic_loss.write(str(info_train["critic_loss"]) + "\n")
				f_critic_loss.flush()

		t1_step = time.time()
		next_observation, reward, done, info = goalsetter.step(
			env, observation, action, *env.step(action)
		)
		t2_step = time.time()
		# print("step time = ", t2_step - t1_step)

		# print("done = ", done)

		# pdb.set_trace()

		step = {
			"observation": observation,
			"action": action,
			"reward": reward,
			"truncation": info["truncation"],
			"done": done,
			"next_observation": next_observation,
		}

		# print("step = ", step)
		# print("info = ", info)

		if env_info["is_goalenv"]:
			step["done_from_env"] = info["done_from_env"]
			step["is_success"] = info["is_success"]
			step["last_skill"] = (info["skill_indx"] == info["next_skill_indx"]).reshape(observation["desired_goal"].shape[0], 1)
			step["next_skill_goal"] = observation["next_skill_goal"].reshape(observation["desired_goal"].shape)

		buffer_.insert(step)

		observation = next_observation.copy()

		t1_reset_time = time.time()
		if done.max():
			traj.append(observation["observation"].copy())

			curr_indx = info["skill_indx"][0][0]
			reset_indx = info["reset_skill_indx"][0][0]

			num_rollouts += 1
			num_rollouts_skill[reset_indx][curr_indx] += 1

			if info["is_success"].max() == 1:
				num_success += 1
				num_success_skill[reset_indx][curr_indx] += 1

			# use store_done() if the buffer is an episodic buffer
			if hasattr(buffer_, "store_done"):
				buffer_.store_done()
			observation = goalsetter.reset_done(env, env.reset_done())
			if len(traj) > 0:
				if info["is_success"].max() == 1:
					s_trajs.append(traj)
				else:
					f_trajs.append(traj)
				traj = []
		t2_reset_time = time.time()
		# print("reset time = ", t2_reset_time - t1_reset_time)

	f_ratio.close()
	f_critic_loss.close()
	f_values.close()
