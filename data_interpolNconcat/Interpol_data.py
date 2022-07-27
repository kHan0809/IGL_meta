import os
import pickle
import numpy as np
from Utils.IGL_interpolate import intpol_pos, intpol_quat

def sub_goal_separator(sub_goal):
    count = 0
    idx_list = []
    for idx in range(len(sub_goal)):
        if sub_goal[idx] != sub_goal[idx+1]:
            count += 1
            idx_list.append(idx)
            if count == 1:
                break
        else:
            pass
    return idx_list

data_concat = []
task_name = 'data_drawer-open-v2-goal-observable'
for pickle_data in os.listdir(os.getcwd()+'/'+os.pardir+'/IGL_data'):
    if task_name+'_0' in pickle_data: #pick-place_mid_sg
        with open('../IGL_data/'+ pickle_data, 'rb') as f:
            data = pickle.load(f)
            data_concat.extend(data)
    else:
        pass

traj_sg0 = []
traj_sg1 = []
traj_sg2 = []
coefs = np.linspace(0,1,3,endpoint=True)
for i in range(len(data_concat)-1):
    for j in range(i+1,len(data_concat)):
        #===================
        cur_robot_pos1 = np.array(data_concat[i]['obs_cur_robot_pos'])
        cur_robot_pos2 = np.array(data_concat[j]['obs_cur_robot_pos'])
        cur_obj_pos1 = np.array(data_concat[i]['obs_cur_obj1_pos'])
        cur_obj_pos2 = np.array(data_concat[j]['obs_cur_obj1_pos'])
        cur_obj_quat1 = np.array(data_concat[i]['obs_cur_obj1_quat'])
        cur_obj_quat2 = np.array(data_concat[j]['obs_cur_obj1_quat'])

        #===================
        pre_robot_pos1 = np.array(data_concat[i]['obs_pre_robot_pos'])
        pre_robot_pos2 = np.array(data_concat[j]['obs_pre_robot_pos'])
        pre_obj_pos1 = np.array(data_concat[i]['obs_pre_obj1_pos'])
        pre_obj_pos2 = np.array(data_concat[j]['obs_pre_obj1_pos'])
        pre_obj_quat1 = np.array(data_concat[i]['obs_pre_obj1_quat'])
        pre_obj_quat2 = np.array(data_concat[j]['obs_pre_obj1_quat'])

        goal1 = np.array(data_concat[i]['goal'])
        goal2 = np.array(data_concat[j]['goal'])

        sub_goal1 = np.array(data_concat[i]['subgoal'])
        sub_goal2 = np.array(data_concat[j]['subgoal'])

        idx_sub_traj1 = sub_goal_separator(sub_goal1)
        idx_sub_traj2 = sub_goal_separator(sub_goal2)


        for k in range(len(idx_sub_traj1)+1):
            if k == 0:
                cur_robot_pos_can1    = cur_robot_pos1[:idx_sub_traj1[k]+1]
                cur_robot_pos_can2    = cur_robot_pos2[:idx_sub_traj2[k]+1]
                cur_obj_pos_can1      = cur_obj_pos1[:idx_sub_traj1[k]+1]
                cur_obj_pos_can2      = cur_obj_pos2[:idx_sub_traj2[k] + 1]
                cur_obj_quat_can1   = cur_obj_quat1[:idx_sub_traj1[k]+1]
                cur_obj_quat_can2   = cur_obj_quat2[:idx_sub_traj2[k]+1]

                pre_robot_pos_can1    = pre_robot_pos1[:idx_sub_traj1[k]+1]
                pre_robot_pos_can2    = pre_robot_pos2[:idx_sub_traj2[k]+1]
                pre_obj_pos_can1      = pre_obj_pos1[:idx_sub_traj1[k]+1]
                pre_obj_pos_can2      = pre_obj_pos2[:idx_sub_traj2[k] + 1]
                pre_obj_quat_can1   = pre_obj_quat1[:idx_sub_traj1[k]+1]
                pre_obj_quat_can2   = pre_obj_quat2[:idx_sub_traj2[k]+1]

                goal_can1           = goal1[:idx_sub_traj1[k] + 1]
                goal_can2           = goal2[:idx_sub_traj2[k] + 1]


            elif k == (len(idx_sub_traj1)):
                cur_robot_pos_can1 = cur_robot_pos1[idx_sub_traj1[k-1]+1:]
                cur_robot_pos_can2 = cur_robot_pos2[idx_sub_traj2[k-1]+1:]
                cur_obj_pos_can1 = cur_obj_pos1[idx_sub_traj1[k-1]+1:]
                cur_obj_pos_can2 = cur_obj_pos2[idx_sub_traj2[k-1]+1:]
                cur_obj_quat_can1 = cur_obj_quat1[idx_sub_traj1[k-1]+1:]
                cur_obj_quat_can2 = cur_obj_quat2[idx_sub_traj2[k-1]+1:]

                pre_robot_pos_can1 = pre_robot_pos1[idx_sub_traj1[k-1]+1:]
                pre_robot_pos_can2 = pre_robot_pos2[idx_sub_traj2[k-1]+1:]
                pre_obj_pos_can1 = pre_obj_pos1[idx_sub_traj1[k-1]+1:]
                pre_obj_pos_can2 = pre_obj_pos2[idx_sub_traj2[k-1]+1:]
                pre_obj_quat_can1 = pre_obj_quat1[idx_sub_traj1[k-1]+1:]
                pre_obj_quat_can2 = pre_obj_quat2[idx_sub_traj2[k-1]+1:]

                goal_can1 = goal1[idx_sub_traj1[k-1]+1:]
                goal_can2 = goal2[idx_sub_traj2[k-1]+1:]
            else:

                cur_robot_pos_can1 = cur_robot_pos1[idx_sub_traj1[k-1]+1:idx_sub_traj1[k]+1]
                cur_robot_pos_can2 = cur_robot_pos2[idx_sub_traj2[k-1]+1:idx_sub_traj2[k]+1]
                cur_obj_pos_can1 = cur_obj_pos1[idx_sub_traj1[k-1]+1:idx_sub_traj1[k]+1]
                cur_obj_pos_can2 = cur_obj_pos2[idx_sub_traj2[k-1]+1:idx_sub_traj2[k]+1]
                cur_obj_quat_can1 = cur_obj_quat1[idx_sub_traj1[k-1]+1:idx_sub_traj1[k]+1]
                cur_obj_quat_can2 = cur_obj_quat2[idx_sub_traj2[k-1]+1:idx_sub_traj2[k]+1]

                pre_robot_pos_can1 = pre_robot_pos1[idx_sub_traj1[k-1]+1:idx_sub_traj1[k]+1]
                pre_robot_pos_can2 = pre_robot_pos2[idx_sub_traj2[k-1]+1:idx_sub_traj2[k]+1]
                pre_obj_pos_can1 = pre_obj_pos1[idx_sub_traj1[k-1]+1:idx_sub_traj1[k]+1]
                pre_obj_pos_can2 = pre_obj_pos2[idx_sub_traj2[k-1]+1:idx_sub_traj2[k]+1]
                pre_obj_quat_can1 = pre_obj_quat1[idx_sub_traj1[k-1]+1:idx_sub_traj1[k]+1]
                pre_obj_quat_can2 = pre_obj_quat2[idx_sub_traj2[k-1]+1:idx_sub_traj2[k]+1]

                goal_can1 = goal1[idx_sub_traj1[k-1]+1:idx_sub_traj1[k]+1]
                goal_can2 = goal2[idx_sub_traj2[k-1]+1:idx_sub_traj2[k]+1]

            if i == 0 and j == 1:
                for coef in coefs:
                    new_cur_robot_pos= intpol_pos(cur_robot_pos_can1,cur_robot_pos_can2,coef)
                    new_cur_obj_pos  = intpol_pos(cur_obj_pos_can1, cur_obj_pos_can2, coef)
                    new_cur_obj_quat = intpol_quat(cur_obj_quat_can1, cur_obj_quat_can2, coef)

                    new_pre_robot_pos= intpol_pos(pre_robot_pos_can1,pre_robot_pos_can2,coef)
                    new_pre_obj_pos  = intpol_pos(pre_obj_pos_can1, pre_obj_pos_can2, coef)
                    new_pre_obj_quat = intpol_quat(pre_obj_quat_can1, pre_obj_quat_can2, coef)

                    new_goal         = intpol_pos(goal_can1,goal_can2,coef)
                    new_sub_goal = np.array(len(new_goal)*[k]).reshape(-1,1)

                    if k == 0:
                        traj_sg0.append(dict(obs_cur_robot_pos=new_cur_robot_pos, obs_cur_obj1_pos=new_cur_obj_pos, obs_cur_obj1_quat=new_cur_obj_quat,\
                                             obs_pre_robot_pos=new_pre_robot_pos, obs_pre_obj1_pos=new_pre_obj_pos, obs_pre_obj1_quat=new_pre_obj_quat,\
                                             goal=new_goal,subgoal=new_sub_goal))
                    elif k==1:
                        traj_sg1.append(dict(obs_cur_robot_pos=new_cur_robot_pos, obs_cur_obj1_pos=new_cur_obj_pos,obs_cur_obj1_quat=new_cur_obj_quat, \
                                             obs_pre_robot_pos=new_pre_robot_pos, obs_pre_obj1_pos=new_pre_obj_pos,obs_pre_obj1_quat=new_pre_obj_quat, \
                                             goal=new_goal, subgoal=new_sub_goal))
                    elif k==2:
                        traj_sg2.append(dict(obs_cur_robot_pos=new_cur_robot_pos, obs_cur_obj1_pos=new_cur_obj_pos,obs_cur_obj1_quat=new_cur_obj_quat, \
                                             obs_pre_robot_pos=new_pre_robot_pos, obs_pre_obj1_pos=new_pre_obj_pos,obs_pre_obj1_quat=new_pre_obj_quat, \
                                             goal=new_goal, subgoal=new_sub_goal))

            elif i == 0 and j != 1:
                for coef in coefs[1:]:
                    new_cur_robot_pos= intpol_pos(cur_robot_pos_can1,cur_robot_pos_can2,coef)
                    new_cur_obj_pos  = intpol_pos(cur_obj_pos_can1, cur_obj_pos_can2, coef)
                    new_cur_obj_quat = intpol_quat(cur_obj_quat_can1, cur_obj_quat_can2, coef)

                    new_pre_robot_pos= intpol_pos(pre_robot_pos_can1,pre_robot_pos_can2,coef)
                    new_pre_obj_pos  = intpol_pos(pre_obj_pos_can1, pre_obj_pos_can2, coef)
                    new_pre_obj_quat = intpol_quat(pre_obj_quat_can1, pre_obj_quat_can2, coef)

                    new_goal         = intpol_pos(goal_can1,goal_can2,coef)
                    new_sub_goal = np.array(len(new_goal)*[k]).reshape(-1,1)

                    if k == 0:
                        traj_sg0.append(dict(obs_cur_robot_pos=new_cur_robot_pos, obs_cur_obj1_pos=new_cur_obj_pos, obs_cur_obj1_quat=new_cur_obj_quat,\
                                             obs_pre_robot_pos=new_pre_robot_pos, obs_pre_obj1_pos=new_pre_obj_pos, obs_pre_obj1_quat=new_pre_obj_quat,\
                                             goal=new_goal,subgoal=new_sub_goal))
                    elif k==1:
                        traj_sg1.append(dict(obs_cur_robot_pos=new_cur_robot_pos, obs_cur_obj1_pos=new_cur_obj_pos,obs_cur_obj1_quat=new_cur_obj_quat, \
                                             obs_pre_robot_pos=new_pre_robot_pos, obs_pre_obj1_pos=new_pre_obj_pos,obs_pre_obj1_quat=new_pre_obj_quat, \
                                             goal=new_goal, subgoal=new_sub_goal))
                    elif k==2:
                        traj_sg2.append(dict(obs_cur_robot_pos=new_cur_robot_pos, obs_cur_obj1_pos=new_cur_obj_pos,obs_cur_obj1_quat=new_cur_obj_quat, \
                                             obs_pre_robot_pos=new_pre_robot_pos, obs_pre_obj1_pos=new_pre_obj_pos,obs_pre_obj1_quat=new_pre_obj_quat, \
                                             goal=new_goal, subgoal=new_sub_goal))

            else:
                for coef in coefs[1:-1]:
                    new_cur_robot_pos= intpol_pos(cur_robot_pos_can1,cur_robot_pos_can2,coef)
                    new_cur_obj_pos  = intpol_pos(cur_obj_pos_can1, cur_obj_pos_can2, coef)
                    new_cur_obj_quat = intpol_quat(cur_obj_quat_can1, cur_obj_quat_can2, coef)

                    new_pre_robot_pos= intpol_pos(pre_robot_pos_can1,pre_robot_pos_can2,coef)
                    new_pre_obj_pos  = intpol_pos(pre_obj_pos_can1, pre_obj_pos_can2, coef)
                    new_pre_obj_quat = intpol_quat(pre_obj_quat_can1, pre_obj_quat_can2, coef)

                    new_goal         = intpol_pos(goal_can1,goal_can2,coef)
                    new_sub_goal = np.array(len(new_goal)*[k]).reshape(-1,1)

                    if k == 0:
                        traj_sg0.append(dict(obs_cur_robot_pos=new_cur_robot_pos, obs_cur_obj1_pos=new_cur_obj_pos, obs_cur_obj1_quat=new_cur_obj_quat,\
                                             obs_pre_robot_pos=new_pre_robot_pos, obs_pre_obj1_pos=new_pre_obj_pos, obs_pre_obj1_quat=new_pre_obj_quat,\
                                             goal=new_goal,subgoal=new_sub_goal))
                    elif k==1:
                        traj_sg1.append(dict(obs_cur_robot_pos=new_cur_robot_pos, obs_cur_obj1_pos=new_cur_obj_pos,obs_cur_obj1_quat=new_cur_obj_quat, \
                                             obs_pre_robot_pos=new_pre_robot_pos, obs_pre_obj1_pos=new_pre_obj_pos,obs_pre_obj1_quat=new_pre_obj_quat, \
                                             goal=new_goal, subgoal=new_sub_goal))
                    elif k==2:
                        traj_sg2.append(dict(obs_cur_robot_pos=new_cur_robot_pos, obs_cur_obj1_pos=new_cur_obj_pos,obs_cur_obj1_quat=new_cur_obj_quat, \
                                             obs_pre_robot_pos=new_pre_robot_pos, obs_pre_obj1_pos=new_pre_obj_pos,obs_pre_obj1_quat=new_pre_obj_quat, \
                                             goal=new_goal, subgoal=new_sub_goal))


print(len(traj_sg0))
print(len(traj_sg1))
print(len(traj_sg2))

with open('../IGL_data/'+task_name+'sg0.pickle', 'wb') as f:
    pickle.dump(traj_sg0, f, pickle.HIGHEST_PROTOCOL)
with open('../IGL_data/'+task_name+'sg1.pickle', 'wb') as f:
    pickle.dump(traj_sg1, f, pickle.HIGHEST_PROTOCOL)
# with open('../IGL_data/'+task_name+'sg0.pickle', 'wb') as f:
#     pickle.dump(traj_sg2, f, pickle.HIGHEST_PROTOCOL)

