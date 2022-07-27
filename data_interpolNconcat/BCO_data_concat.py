import pickle
import numpy as np
import os

data_concat = []

data_concat = []
task_name = 'data_drawer-open-v2-goal-observable'
for pickle_data in os.listdir(os.getcwd()+'/'+os.pardir+'/IGL_data'):
    if task_name+'_0' in pickle_data:
        with open('../IGL_data/'+ pickle_data, 'rb') as f:
            data = pickle.load(f)
            data_concat.extend(data)
    else:
        pass

BCO_state = []

print(len(data_concat))
for traj in data:
    for i in range(len(traj["obs_cur_robot_pos"])):
        try:
            BCO_state.append(
                np.concatenate((traj["obs_cur_robot_pos"][i], traj["obs_cur_obj1_pos"][i], traj["obs_cur_obj1_quat"][i], \
                                traj["obs_pre_robot_pos"][i], traj["obs_pre_obj1_pos"][i], traj["obs_pre_obj1_quat"][i], \
                                traj["goal"][i]
                                )))
        except:
            BCO_state.append(
                np.concatenate((traj["obs_cur_robot_pos"][i], traj["obs_cur_obj1_pos"][i], traj["obs_cur_obj1_quat"][i], \
                                traj["obs_pre_robot_pos"][i], traj["obs_pre_obj1_pos"][i], traj["obs_pre_obj1_quat"][i], \
                                traj["goal"][i]
                                )))

BCO_state = np.array(BCO_state)
print(BCO_state.shape)

np.save('../IGL_data/DrawerOpen_BCO_demo',BCO_state)