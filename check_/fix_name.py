import os
import pickle
data_concat = []
for pickle_data in os.listdir(os.getcwd()+'/IGL_data'):
    if 'data_' in pickle_data:
        with open('./IGL_data/'+ pickle_data, 'rb') as f:
            data = pickle.load(f)
            data_concat.extend(data)
    else:
        pass

traj_sg0 = []
traj_sg1 = []
traj_sg2 = []
traj_sg3 = []
print(len(data_concat))
total_epi = []

for data in data_concat:
  total_epi.append(dict(obs_cur_robot_pos=data['epi_obs_cur_robot_pos'],obs_cur_obj1_pos=data['epi_obs_cur_obj1_pos'],obs_cur_obj1_quat=data['epi_obs_cur_obj1_quat'],\
                        obs_pre_robot_pos=data['epi_obs_pre_robot_pos'],obs_pre_obj1_pos=data['epi_obs_pre_obj1_pos'],obs_pre_obj1_quat=data['epi_obs_pre_obj1_quat'],\
                        goal=data['epi_goal'],subgoal=data['epi_subgoal'],action=data['epi_action']))


with open('../IGL_data/data_total.pickle', 'wb') as f:
  pickle.dump(total_epi, f, pickle.HIGHEST_PROTOCOL)