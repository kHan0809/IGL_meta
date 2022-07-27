import metaworld
import random
import numpy as np
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
import pickle
from Utils.utils import obs2dictobs, human_key_control , get_subgoal


def get_epi(task_name):
  task_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task_name]
  env = task_observable_cls()

  epi_obs_cur_robot_pos, epi_obs_cur_obj1_pos, epi_obs_cur_obj1_quat, epi_obs_pre_robot_pos, epi_obs_pre_obj1_pos, epi_obs_pre_obj1_quat, epi_goal     = [], [], [], [], [], [], []
  epi_action, epi_subgoal  = [], []

  obs = env.reset()
  dictobs = obs2dictobs(obs)
  subgoal = get_subgoal(dictobs,np.array([0]),task_name)
  #======================================================
  epi_obs_cur_robot_pos.append(dictobs['obs_cur_robot_pos'])
  epi_obs_cur_obj1_pos.append(dictobs['obs_cur_obj1_pos'])
  epi_obs_cur_obj1_quat.append(dictobs['obs_cur_obj1_quat'])
  epi_obs_pre_robot_pos.append(dictobs['obs_pre_robot_pos'])
  epi_obs_pre_obj1_pos.append(dictobs['obs_pre_obj1_pos'])
  epi_obs_pre_obj1_quat.append(dictobs['obs_pre_obj1_quat'])
  epi_goal.append(dictobs['goal'])
  epi_subgoal.append(subgoal)

  success_count = 0
  for i in range(2000):
    try:
      a = human_key_control(input())
    except:
      a = human_key_control(input())

    obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
    epi_action.append(a)

    dictobs = obs2dictobs(obs)
    subgoal = get_subgoal(dictobs, subgoal, task_name)
    epi_obs_cur_robot_pos.append(dictobs['obs_cur_robot_pos'])
    epi_obs_cur_obj1_pos.append(dictobs['obs_cur_obj1_pos'])
    epi_obs_cur_obj1_quat.append(dictobs['obs_cur_obj1_quat'])
    epi_obs_pre_robot_pos.append(dictobs['obs_pre_robot_pos'])
    epi_obs_pre_obj1_pos.append(dictobs['obs_pre_obj1_pos'])
    epi_obs_pre_obj1_quat.append(dictobs['obs_pre_obj1_quat'])
    epi_goal.append(dictobs['goal'])
    epi_subgoal.append(subgoal)
    print("=====================")
    print(subgoal)
    print(dictobs['obs_cur_robot_pos'][:-1] - dictobs['obs_cur_obj1_pos'],dictobs['obs_cur_robot_pos'][-1])
    env.render()
    if info['success']:
      success_count += 1
      if success_count >= 10:
        env.close()
        break
  episode = dict(obs_cur_robot_pos = epi_obs_cur_robot_pos, obs_cur_obj1_pos=epi_obs_cur_obj1_pos, obs_cur_obj1_quat=epi_obs_cur_obj1_quat,\
                 obs_pre_robot_pos = epi_obs_pre_robot_pos, obs_pre_obj1_pos=epi_obs_pre_obj1_pos, obs_pre_obj1_quat=epi_obs_pre_obj1_quat, goal=epi_goal, subgoal=epi_subgoal,action=epi_action)
  return episode
if __name__ == "__main__":
  # "box-close-v2-goal-observable" "drawer-open-v2-goal-observable" "drawer-close-v2-goal-observable" "pick-place-v2-goal-observable"
  task_name = "drawer-open-v2-goal-observable"
  file_num = str(1)
  traj_num = 0
  total_epi = []
  while True:
    epi = get_epi(task_name)
    total_epi.append(epi)
    traj_num += 1
    print(traj_num)
    with open('./IGL_data/data_'+task_name+'_'+file_num+'.pickle', 'wb') as f:
      pickle.dump(total_epi, f, pickle.HIGHEST_PROTOCOL)