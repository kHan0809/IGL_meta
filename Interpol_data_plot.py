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
            if count == 2:
                break
        else:
            pass
    return idx_list


data_concat = []
subgoal = '2'
for pickle_data in os.listdir(os.getcwd()+'/IGL_data'):
    if 'pick-place_mid_sg'+subgoal in pickle_data: #pick-place_mid_sg

        with open('./IGL_data/'+ pickle_data, 'rb') as f:
            data = pickle.load(f)
            data_concat.extend(data)
    else:
        pass

All_traj = []
print(len(data_concat))
coefs = np.linspace(0,1,4,endpoint=True)

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection="3d")

for i in range(200):
    #===================
    cur_robot_pos1 = np.array(data_concat[i]['obs_cur_robot_pos'])
    cur_obj_pos1 = np.array(data_concat[i]['obs_cur_obj1_pos'])
    cur_obj_quat1 = np.array(data_concat[i]['obs_cur_obj1_quat'])


    #===================
    pre_robot_pos1 = np.array(data_concat[i]['obs_pre_robot_pos'])
    pre_obj_pos1 = np.array(data_concat[i]['obs_pre_obj1_pos'])
    pre_obj_quat1 = np.array(data_concat[i]['obs_pre_obj1_quat'])

    goal1 = np.array(data_concat[i]['goal'])
    sub_goal1 = np.array(data_concat[i]['subgoal'])


    x,y,z,grip = zip(*cur_robot_pos1)
    # Creating plot
    ax.scatter3D(x, y, z, color="r")


    # x, y, z, grip = zip(*cur_robot_pos_can2)
    # # Creating plot
    # ax.scatter3D(x, y, z, color="b")
    #
    # #==========obj============
    # x, y, z = zip(*cur_obj_pos_can1)
    # # Creating plot
    # ax.scatter3D(x, y, z, color="k")

    # x,y,z = zip(*new_cur_obj_pos)
    # print(cur_obj_pos_can1)
    # Creating plot
    # ax.scatter3D(x, y, z, color="m")

defal = 0.15

ax.set_xlim([-defal + x[len(x)//2], defal + x[len(x)//2]])
ax.set_ylim([-defal + y[len(x)//2], defal + y[len(x)//2]])
ax.set_zlim([-defal + z[len(x)//2], defal + z[len(x)//2]])
ax.set_xlabel('X___')
ax.set_ylabel('Y___')
ax.set_zlabel('Z___')



plt.show()
raise




# print(len(All_traj))
# with open('./IGL_data/using_mid'+subgoal+'.pickle', 'wb') as f:
#     pickle.dump(All_traj, f, pickle.HIGHEST_PROTOCOL)

