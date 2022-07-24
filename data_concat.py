import pickle
import numpy as np
import os

data_concat = []
subgoal = '1'
for pickle_data in os.listdir(os.getcwd()+'/IGL_data'):
    if 'pick-place_mid_sg'+subgoal in pickle_data:
        with open('./IGL_data/'+ pickle_data, 'rb') as f:
            data = pickle.load(f)
            data_concat.extend(data)
    else:
        pass

new_x = []
new_y = []
new_x_import = []
new_y_import = []
for traj in data:
    for i in range(len(traj["obs_cur_robot_pos"])-1):
        if i < (len(traj["obs_cur_robot_pos"])-1)*0.9:
            new_x.append(np.concatenate((traj["obs_cur_robot_pos"][i], traj["obs_cur_obj1_pos"][i] ,traj["obs_cur_obj1_quat"][i],\
                            traj["obs_pre_robot_pos"][i], traj["obs_pre_obj1_pos"][i] ,traj["obs_pre_obj1_quat"][i],\
                            traj["goal"][i],traj["subgoal"][i]
                            )))
            new_y.append(traj["obs_cur_robot_pos"][i+1])
        else:
            new_x_import.append(np.concatenate((traj["obs_cur_robot_pos"][i], traj["obs_cur_obj1_pos"][i] ,traj["obs_cur_obj1_quat"][i],\
                            traj["obs_pre_robot_pos"][i], traj["obs_pre_obj1_pos"][i] ,traj["obs_pre_obj1_quat"][i],\
                            traj["goal"][i],traj["subgoal"][i]
                            )))
            new_y_import.append(traj["obs_cur_robot_pos"][i+1])

np_x = np.array(new_x)
np_y = np.array(new_y)

np_x_imp = np.array(new_x_import)
np_y_imp = np.array(new_y_import)

np.save('./IGL_data/np_x_sg'+subgoal+'_no_imp',np_x)
np.save('./IGL_data/np_y_sg'+subgoal+'_no_imp',np_y)
np.save('./IGL_data/np_x_sg'+subgoal+'_imp',np_x_imp)
np.save('./IGL_data/np_y_sg'+subgoal+'_imp',np_y_imp)