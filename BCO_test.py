import metaworld
import numpy as np
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
from Utils.utils import obs2dictobs, obs2BCO_state , get_subgoal
from Utils.arguments import get_args
from Model.class_model import BCO
import torch

args = get_args()
# "box-close-v2-goal-observable" "drawer-open-v2-goal-observable" "drawer-close-v2-goal-observable"
task_name = args.task_name
task_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task_name]
env = task_observable_cls()
eval_env = task_observable_cls()

obs = env.reset()
bco_state = obs2BCO_state(obs)

state_dim = args.BCO_state_dim

expert_demo = np.load('./IGL_data/DrawerOpen_BCO_demo.npy')
bco = BCO(state_dim, env.action_space.sample().shape[0], expert_demo, args)
bco.load_state_dict(torch.load("./model_save/bco_model1_1000.pt"))

epi_return = []
epi_success = []

eval_obs = eval_env.reset()
eval_bco_state = obs2BCO_state(eval_obs)
bco.eval()
while True:
  for eval_epi in range(args.BCO_eval_epi):
    Return = 0
    eval_success_count = 0
    eval_success = False
    for step in range(args.BCO_epi_len):
      eval_action = bco.BC(torch.FloatTensor(eval_bco_state).unsqueeze(0).to(args.device_train)).squeeze(
        0).cpu().detach().numpy()
      eval_next_obs, reward, done, info = eval_env.step(eval_action)
      eval_bco_state = obs2BCO_state(eval_next_obs)
      Return += reward

      if info['success']:
        eval_success_count += 1
        if eval_success_count >= 5:
          eval_success = True
          break
      eval_env.render()

    eval_env.close()
    task_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task_name]
    eval_env = task_observable_cls()
    eval_obs = eval_env.reset()
    eval_bco_state = obs2BCO_state(eval_obs)

    epi_success.append(eval_success)
    epi_return.append(Return)
  print("==================[Eval]====================")
  print("Mean return  : ", np.mean(epi_return), "Min return", np.min(epi_return), "Max return", np.max(epi_return))
  print("Success Rate : ", np.mean(epi_success))
  print("============================================")