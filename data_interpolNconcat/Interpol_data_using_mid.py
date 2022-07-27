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
subgoal = '1'
task_name = 'data_drawer-open-v2-goal-observable'
for pickle_data in os.listdir(os.getcwd()+'/'+os.pardir+'/IGL_data'):
    if task_name+'sg'+subgoal in pickle_data:
        with open('../IGL_data/' + pickle_data, 'rb') as f:
            data = pickle.load(f)
            data_concat.extend(data)
    else:
        pass

All_traj = []
coefs = np.linspace(0,1,4,endpoint=True)

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


        cur_robot_pos_can1 = cur_robot_pos1
        cur_robot_pos_can2 = cur_robot_pos2
        cur_obj_pos_can1   = cur_obj_pos1
        cur_obj_pos_can2   = cur_obj_pos2
        cur_obj_quat_can1  = cur_obj_quat1
        cur_obj_quat_can2 = cur_obj_quat2

        pre_robot_pos_can1 = pre_robot_pos1
        pre_robot_pos_can2 = pre_robot_pos2
        pre_obj_pos_can1   = pre_obj_pos1
        pre_obj_pos_can2   = pre_obj_pos2
        pre_obj_quat_can1  = pre_obj_quat1
        pre_obj_quat_can2  = pre_obj_quat2

        goal_can1 = goal1
        goal_can2 = goal2


        if i == 0 and j == 1:
            for coef in coefs:
                new_cur_robot_pos= intpol_pos(cur_robot_pos_can1,cur_robot_pos_can2,coef)
                new_cur_obj_pos  = intpol_pos(cur_obj_pos_can1, cur_obj_pos_can2, coef)
                new_cur_obj_quat = intpol_quat(cur_obj_quat_can1, cur_obj_quat_can2, coef)

                new_pre_robot_pos= intpol_pos(pre_robot_pos_can1,pre_robot_pos_can2,coef)
                new_pre_obj_pos  = intpol_pos(pre_obj_pos_can1, pre_obj_pos_can2, coef)
                new_pre_obj_quat = intpol_quat(pre_obj_quat_can1, pre_obj_quat_can2, coef)

                new_goal         = intpol_pos(goal_can1,goal_can2,coef)
                new_sub_goal = np.array(len(new_goal)*[sub_goal1[0]]).reshape(-1,1)

                All_traj.append(dict(obs_cur_robot_pos=new_cur_robot_pos, obs_cur_obj1_pos=new_cur_obj_pos,obs_cur_obj1_quat=new_cur_obj_quat, \
                                         obs_pre_robot_pos=new_pre_robot_pos, obs_pre_obj1_pos=new_pre_obj_pos,obs_pre_obj1_quat=new_pre_obj_quat, \
                                         goal=new_goal, subgoal=new_sub_goal))

        elif i == 0 and j != 1:
            for coef in coefs[1:]:
                new_cur_robot_pos = intpol_pos(cur_robot_pos_can1, cur_robot_pos_can2, coef)
                new_cur_obj_pos = intpol_pos(cur_obj_pos_can1, cur_obj_pos_can2, coef)
                new_cur_obj_quat = intpol_quat(cur_obj_quat_can1, cur_obj_quat_can2, coef)

                new_pre_robot_pos = intpol_pos(pre_robot_pos_can1, pre_robot_pos_can2, coef)
                new_pre_obj_pos = intpol_pos(pre_obj_pos_can1, pre_obj_pos_can2, coef)
                new_pre_obj_quat = intpol_quat(pre_obj_quat_can1, pre_obj_quat_can2, coef)

                new_goal = intpol_pos(goal_can1, goal_can2, coef)
                new_sub_goal = np.array(len(new_goal) * [sub_goal1[0]]).reshape(-1, 1)

                All_traj.append(dict(obs_cur_robot_pos=new_cur_robot_pos, obs_cur_obj1_pos=new_cur_obj_pos,
                                     obs_cur_obj1_quat=new_cur_obj_quat, \
                                     obs_pre_robot_pos=new_pre_robot_pos, obs_pre_obj1_pos=new_pre_obj_pos,
                                     obs_pre_obj1_quat=new_pre_obj_quat, \
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
                new_sub_goal = np.array(len(new_goal)*[sub_goal1[0]]).reshape(-1,1)

                All_traj.append(dict(obs_cur_robot_pos=new_cur_robot_pos, obs_cur_obj1_pos=new_cur_obj_pos,obs_cur_obj1_quat=new_cur_obj_quat, \
                                         obs_pre_robot_pos=new_pre_robot_pos, obs_pre_obj1_pos=new_pre_obj_pos,obs_pre_obj1_quat=new_pre_obj_quat, \
                                         goal=new_goal, subgoal=new_sub_goal))


print(len(All_traj))
with open('../IGL_data/'+task_name+'_using_mid_'+subgoal+'.pickle', 'wb') as f:
    pickle.dump(All_traj, f, pickle.HIGHEST_PROTOCOL)

