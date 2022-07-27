import metaworld
import numpy as np
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
from Utils.utils import obs2dictobs, human_key_control , get_subgoal, obs2igl_state
from Model.model import hBC
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
  hbc = hBC(all_dim,device)
  hbc.load_state_dict(torch.load('./model_save/hBC/DrawerOpenhBC0120'))

  hbc.eval()
  while True:
    success_count = 0
    for i in range(500):
      one_state = obs2igl_state(obs,subgoal)
      if subgoal == 0:
        action=hbc(torch.FloatTensor(one_state).unsqueeze(0)).squeeze(0).detach().numpy()
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





