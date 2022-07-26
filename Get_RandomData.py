import metaworld
import random
import numpy as np
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
import pickle
from Utils.utils import obs2dictobs, human_key_control , get_subgoal
# print(metaworld.ML1.ENV_NAMES)  # Check out the available environments


def get_epi(task_name):
  task_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task_name]
  env = task_observable_cls()

  epi_obs_cur, epi_obs_next, epi_action  = [], [], []

  obs = env.reset()
  dictobs = obs2dictobs(obs)
  #======================================================
  print(env.action_space)
  raise
  for i in range(2000):
    epi_obs_cur.append(dictobs['obs_cur_robot_pos'])
    a = env.action_space.sample()
    obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
    epi_action.append(a)
    dictobs = obs2dictobs(obs)
    epi_obs_next.append(dictobs['obs_cur_robot_pos'])

    # print("=====================")
    # print(dictobs['obs_cur_robot_pos'][:-1] - dictobs['obs_cur_obj1_pos'],dictobs['obs_cur_robot_pos'][-1])
    # env.render()

  episode = dict(obs_cur = epi_obs_cur, action=epi_action, obs_next=epi_obs_next)
  return episode
if __name__ == "__main__":
  # "box-close-v2-goal-observable" "drawer-open-v2-goal-observable" "drawer-close-v2-goal-observable"
  task_name = "pick-place-v2-goal-observable"
  traj_num = 0
  total_epi = []
  count = 0
  while count<500:
    epi = get_epi(task_name)

    total_epi.append(epi)
    count += 1
    print(count)
  with open('./IGL_data/data_random' + '.pickle', 'wb') as f:
    pickle.dump(total_epi, f, pickle.HIGHEST_PROTOCOL)