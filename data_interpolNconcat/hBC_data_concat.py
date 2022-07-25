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

new_x_sg0 = []
new_y_sg0 = []
new_x_sg1 = []
new_y_sg1 = []
new_x_sg2 = []
new_y_sg2 = []

print(len(data_concat))
for traj in data:
    for i in range(len(traj["obs_cur_robot_pos"])-1):
        if traj["subgoal"][i] == 0:
            try:
                new_x_sg0.append(
                    np.concatenate((traj["obs_cur_robot_pos"][i], traj["obs_cur_obj1_pos"][i], traj["obs_cur_obj1_quat"][i], \
                                    traj["obs_pre_robot_pos"][i], traj["obs_pre_obj1_pos"][i], traj["obs_pre_obj1_quat"][i], \
                                    traj["goal"][i], np.array([traj["subgoal"][i]])
                                    )))
            except:
                new_x_sg0.append(
                    np.concatenate((traj["obs_cur_robot_pos"][i], traj["obs_cur_obj1_pos"][i], traj["obs_cur_obj1_quat"][i], \
                                    traj["obs_pre_robot_pos"][i], traj["obs_pre_obj1_pos"][i], traj["obs_pre_obj1_quat"][i], \
                                    traj["goal"][i], traj["subgoal"][i]
                                    )))

            new_y_sg0.append(traj["action"][i])
        if traj["subgoal"][i] == 1:
            try:
                new_x_sg1.append(
                    np.concatenate(
                        (traj["obs_cur_robot_pos"][i], traj["obs_cur_obj1_pos"][i], traj["obs_cur_obj1_quat"][i], \
                         traj["obs_pre_robot_pos"][i], traj["obs_pre_obj1_pos"][i], traj["obs_pre_obj1_quat"][i], \
                         traj["goal"][i], np.array([traj["subgoal"][i]])
                         )))
            except:
                new_x_sg1.append(
                    np.concatenate(
                        (traj["obs_cur_robot_pos"][i], traj["obs_cur_obj1_pos"][i], traj["obs_cur_obj1_quat"][i], \
                         traj["obs_pre_robot_pos"][i], traj["obs_pre_obj1_pos"][i], traj["obs_pre_obj1_quat"][i], \
                         traj["goal"][i], traj["subgoal"][i]
                         )))

            new_y_sg1.append(traj["action"][i])
        if traj["subgoal"][i] == 2:
            try:
                new_x_sg2.append(
                    np.concatenate(
                        (traj["obs_cur_robot_pos"][i], traj["obs_cur_obj1_pos"][i], traj["obs_cur_obj1_quat"][i], \
                         traj["obs_pre_robot_pos"][i], traj["obs_pre_obj1_pos"][i], traj["obs_pre_obj1_quat"][i], \
                         traj["goal"][i], np.array([traj["subgoal"][i]])
                         )))
            except:
                new_x_sg2.append(
                    np.concatenate(
                        (traj["obs_cur_robot_pos"][i], traj["obs_cur_obj1_pos"][i], traj["obs_cur_obj1_quat"][i], \
                         traj["obs_pre_robot_pos"][i], traj["obs_pre_obj1_pos"][i], traj["obs_pre_obj1_quat"][i], \
                         traj["goal"][i], traj["subgoal"][i]
                         )))

            new_y_sg2.append(traj["action"][i])

np_x_sg0 = np.array(new_x_sg0)
np_y_sg0 = np.array(new_y_sg0)

np_x_sg1 = np.array(new_x_sg1)
np_y_sg1 = np.array(new_y_sg1)

np_x_sg2 = np.array(new_x_sg2)
np_y_sg2 = np.array(new_y_sg2)

print(np_x_sg0.shape)
print(np_y_sg0.shape)
print(np_x_sg1.shape)
print(np_y_sg1.shape)
print(np_x_sg2.shape)
print(np_y_sg2.shape)


np.save('../IGL_data/hBC_x_sg0',np_x_sg0)
np.save('../IGL_data/hBC_y_sg0',np_y_sg0)
np.save('../IGL_data/hBC_x_sg1',np_x_sg1)
np.save('../IGL_data/hBC_y_sg1',np_y_sg1)
np.save('../IGL_data/hBC_x_sg2',np_x_sg2)
np.save('../IGL_data/hBC_y_sg2',np_y_sg2)