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
subgoal = '0'
for pickle_data in os.listdir(os.getcwd()+'/IGL_data'):
    if 'pick-place_mid_sg'+subgoal in pickle_data:
        with open('./IGL_data/'+ pickle_data, 'rb') as f:
            data = pickle.load(f)
            data_concat.extend(data)
    else:
        pass

All_traj = []
print(len(data_concat))
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

        print(cur_obj_pos1[:-1]-pre_obj_pos1[1:])
        raise

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

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 7))
        ax = plt.axes(projection="3d")


        for coef in coefs:
            new_cur_robot_pos= intpol_pos(cur_robot_pos_can1,cur_robot_pos_can2,coef)
            new_cur_obj_pos  = intpol_pos(cur_obj_pos_can1, cur_obj_pos_can2, coef)
            new_cur_obj_quat = intpol_quat(cur_obj_quat_can1, cur_obj_quat_can2, coef)
            #============plot=============


            x,y,z,grip=zip(*new_cur_robot_pos)
            # Creating plot
            ax.scatter3D(x, y, z, color="c")


            x,y,z,grip = zip(*cur_robot_pos_can1)
            # Creating plot
            ax.scatter3D(x, y, z, color="r")


            x, y, z, grip = zip(*cur_robot_pos_can2)
            # Creating plot
            ax.scatter3D(x, y, z, color="b")

            #==========obj============
            x, y, z = zip(*cur_obj_pos_can1)
            # Creating plot
            ax.scatter3D(x, y, z, color="k")

            x,y,z = zip(*new_cur_obj_pos)
            print(cur_obj_pos_can1)
            # Creating plot
            ax.scatter3D(x, y, z, color="m")

            defal = 0.15

            ax.set_xlim([-defal + x[len(x)//2], defal + x[len(x)//2]])
            ax.set_ylim([-defal + y[len(x)//2], defal + y[len(x)//2]])
            ax.set_zlim([-defal + z[len(x)//2], defal + z[len(x)//2]])
            ax.set_xlabel('X___')
            ax.set_ylabel('Y___')
            ax.set_zlabel('Z___')



            # show plot



            new_pre_robot_pos= intpol_pos(pre_robot_pos_can1,pre_robot_pos_can2,coef)
            new_pre_obj_pos  = intpol_pos(pre_obj_pos_can1, pre_obj_pos_can2, coef)
            new_pre_obj_quat = intpol_quat(pre_obj_quat_can1, pre_obj_quat_can2, coef)

            new_goal         = intpol_pos(goal_can1,goal_can2,coef)




            new_sub_goal = np.array(len(new_goal)*[sub_goal1[0]]).reshape(-1,1)

        plt.show()
        raise




# print(len(All_traj))
# with open('./IGL_data/using_mid'+subgoal+'.pickle', 'wb') as f:
#     pickle.dump(All_traj, f, pickle.HIGHEST_PROTOCOL)

