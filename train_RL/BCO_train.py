import metaworld
import numpy as np
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
from Utils.utils import obs2dictobs, obs2BCO_state , get_subgoal
from Utils.arguments import get_args
from Model.class_model import BCO
import torch


if __name__ == "__main__":
  for iter in range(5):
    args = get_args()
    # "box-close-v2-goal-observable" "drawer-open-v2-goal-observable" "drawer-close-v2-goal-observable"
    task_name = args.task_name
    task_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task_name]
    env      = task_observable_cls()
    eval_env = task_observable_cls()

    obs = env.reset()
    bco_state = obs2BCO_state(obs)

    state_dim = args.BCO_state_dim

    expert_demo = np.load('../IGL_data/DrawerOpen_BCO_demo.npy')
    bco = BCO(state_dim,env.action_space.sample().shape[0],expert_demo,args)
    # bco.load_state_dict(torch.load("../model_save/bco_model.pt"))

    f = open("../Result/BCO_result"+str(iter)+".txt", 'w')
    f.close


    for episode in range(args.BCO_T):
      success_count = 0
      bco.BC.eval()
      for step in range(args.BCO_epi_len):
        action = bco.BC(torch.FloatTensor(bco_state).unsqueeze(0).to(args.device_train)).squeeze(0).cpu().detach().numpy()

        next_obs,reward,done,info = env.step(action)
        bco_next_state = obs2BCO_state(next_obs)
        bco.Inv_buffer.store_sample(bco_state,action,reward,bco_next_state,done)

        bco_state = bco_next_state

        if info['success']:
          success_count += 1
          if success_count >= 5:
            break

      #===train===
      bco.Inv_train()
      bco.BC_train()
      #===reset env===
      env.close()
      task_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task_name]
      env = task_observable_cls()

      obs = env.reset()
      bco_state = obs2BCO_state(obs)
      # ==============eval=================
      if episode%5 == 0:
        bco.BC.eval()
        epi_return  = []
        epi_success = []

        eval_obs = eval_env.reset()
        eval_bco_state = obs2BCO_state(eval_obs)
        for eval_epi in range(args.BCO_eval_epi):
          Return = 0
          eval_success_count = 0
          eval_success = False
          for step in range(args.BCO_epi_len):
            eval_action = bco.BC(torch.FloatTensor(eval_bco_state).unsqueeze(0).to(args.device_train)).squeeze(0).cpu().detach().numpy()
            eval_next_obs, reward, done, info = eval_env.step(eval_action)
            eval_bco_state = obs2BCO_state(eval_next_obs)
            Return += reward

            if info['success']:
              eval_success_count += 1
              if eval_success_count >= 5:
                eval_success = True
                break

          eval_env.close()
          task_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task_name]
          eval_env = task_observable_cls()
          eval_obs = eval_env.reset()
          eval_bco_state = obs2BCO_state(eval_obs)

          epi_success.append(eval_success)
          epi_return.append(Return)
        print("==================[Eval]====================")
        print("Mean return  : ", np.mean(epi_return),"Min return",np.min(epi_return),"Max return",np.max(epi_return))
        print("Success Rate : ", np.mean(epi_success))
        print("============================================")
        f = open("../Result/BCO_result"+str(iter)+".txt", 'a')
        f.write(str(int(args.BCO_epi_len*(episode+1))))
        f.write(" ")
        f.write(str(round(np.mean(epi_return), 2)))
        f.write(" ")
        f.write(str(round(np.min(epi_return), 2)))
        f.write(" ")
        f.write(str(round(np.max(epi_return), 2)))
        f.write(" ")
        f.write(str(round(np.mean(epi_success), 2)))
        f.write("\n")
        f.close()

      if episode%100 == 99:
        torch.save(bco.state_dict(), "../model_save/bco_model"+str(iter)+"_"+str(episode+1)+".pt")











