import metaworld
import numpy as np
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
from Utils.utils import obs2dictobs, human_key_control , get_subgoal, obs2igl_state
from Model.model import IGL, InvKin
import torch
# print(metaworld.ML1.ENV_NAMES)  # Check out the available environments

if __name__ == "__main__":
  # "box-close-v2-goal-observable" "drawer-open-v2-goal-observable" "drawer-close-v2-goal-observable"
  task_name = "drawer-open-v2-goal-observable"
  task_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task_name]
  env = task_observable_cls()

  obs = env.reset()
  dictobs = obs2dictobs(obs)
  subgoal = get_subgoal(dictobs,np.array([0]),task_name)

  all_dim = 26
  device = "cpu"
  igl0 = IGL(all_dim,device)
  igl0.load_state_dict(torch.load('./model_save/Min/DrawerOpen_Min0121'))
  igl1 = IGL(all_dim, device)
  igl1.load_state_dict(torch.load('./model_save/Min/DrawerOpen_Min1100'))
  # igl0 = IGL(all_dim,device)
  # igl0.load_state_dict(torch.load('./model_save/only_IGL/DrawerOpen0noimp010'))
  # igl1 = IGL(all_dim, device)
  # igl1.load_state_dict(torch.load('./model_save/only_IGL/DrawerOpen1noimp111'))

  igl0.eval()
  igl1.eval()

  while True:
    success_count = 0
    for i in range(500):
      one_state = obs2igl_state(obs,subgoal)
      print(subgoal)
      if subgoal == 0:
        next = igl0(torch.FloatTensor(one_state).unsqueeze(0)).squeeze(0).detach().numpy()
      if subgoal == 1:
        next = igl1(torch.FloatTensor(one_state).unsqueeze(0)).squeeze(0).detach().numpy()

      action=(next-obs[:4])*30


      # action[1] *= 5
      action[-1] *= -1
      print("=============")
      print(obs[:4])
      print(action)
      obs,reward,done,info = env.step(action)

      dictobs=obs2dictobs(obs)
      subgoal = get_subgoal(dictobs,subgoal,task_name)

      env.render()

      if info['success']:
        success_count += 1
        if success_count >= 10:
          env.close()
          break
    env.close()
    task_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task_name]
    env = task_observable_cls()

    obs = env.reset()
    dictobs = obs2dictobs(obs)
    subgoal = get_subgoal(dictobs, np.array([0]),task_name)





