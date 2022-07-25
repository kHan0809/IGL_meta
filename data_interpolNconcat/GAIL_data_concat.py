import pickle
import numpy as np
import os


data_concat = []
for pickle_data in os.listdir(os.getcwd()+'/'+os.pardir+'/IGL_data'):
    if 'data_total'in pickle_data:
        with open('../IGL_data/'+ pickle_data, 'rb') as f:
            data = pickle.load(f)
            data_concat.extend(data)
    else:
        pass

new_x_sg = []
new_y_sg = []

print(len(data_concat))
for traj in data:
    for i in range(len(traj["obs_cur_robot_pos"])-1):
        try:
            new_x_sg.append(
                np.concatenate((traj["obs_cur_robot_pos"][i], traj["obs_cur_obj1_pos"][i], traj["obs_cur_obj1_quat"][i], \
                                traj["obs_pre_robot_pos"][i], traj["obs_pre_obj1_pos"][i], traj["obs_pre_obj1_quat"][i], \
                                traj["goal"][i], np.array([traj["subgoal"][i]])
                                )))
        except:
            new_x_sg.append(
                np.concatenate((traj["obs_cur_robot_pos"][i], traj["obs_cur_obj1_pos"][i], traj["obs_cur_obj1_quat"][i], \
                                traj["obs_pre_robot_pos"][i], traj["obs_pre_obj1_pos"][i], traj["obs_pre_obj1_quat"][i], \
                                traj["goal"][i], traj["subgoal"][i]
                                )))

        new_y_sg.append(traj["action"][i])
np_x_sg = np.array(new_x_sg)
np_y_sg = np.array(new_y_sg)


np.save('../IGL_data/GAIL_x',np_x_sg)
np.save('../IGL_data/GAIL_y',np_y_sg)