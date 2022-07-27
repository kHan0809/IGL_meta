import pickle
import numpy as np
import os

data_concat = []
subgoal = '0'
task_name = 'data_drawer-open-v2-goal-observable'
for pickle_data in os.listdir(os.getcwd()+'/'+os.pardir+'/IGL_data'):
    if task_name+'_using_mid_'+subgoal in pickle_data:
        with open('../IGL_data/'+ pickle_data, 'rb') as f:
            data = pickle.load(f)
            data_concat.extend(data)
    else:
        pass

new_x = []
new_y = []
for traj in data:
    for i in range(len(traj["obs_cur_robot_pos"])-1):
        new_x.append(np.concatenate((traj["obs_cur_robot_pos"][i], traj["obs_cur_obj1_pos"][i] ,traj["obs_cur_obj1_quat"][i],\
                        traj["obs_pre_robot_pos"][i], traj["obs_pre_obj1_pos"][i] ,traj["obs_pre_obj1_quat"][i],\
                        traj["goal"][i],traj["subgoal"][i]
                        )))
        new_y.append(traj["obs_cur_robot_pos"][i+1])

np_x = np.array(new_x)
np_y = np.array(new_y)

print(np_x.shape)
print(np_y.shape)

np.save('../IGL_data/DrawerOpen_x_sg'+subgoal+'_small',np_x)
np.save('../IGL_data/DrawerOpen_y_sg'+subgoal+'_small',np_y)