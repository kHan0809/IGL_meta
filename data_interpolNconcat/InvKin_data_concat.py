import pickle
import numpy as np
import os


data_concat = []
for pickle_data in os.listdir(os.getcwd()+'/'+os.pardir+'/IGL_data'):
    if 'data_random' in pickle_data:
        with open('../IGL_data/'+ pickle_data, 'rb') as f:
            data = pickle.load(f)
            data_concat.extend(data)
    else:
        pass

new_x = []
new_y = []

print(len(data_concat))
for traj in data:
    for i in range(len(traj["obs_cur"])):
        new_x.append(np.concatenate((traj["obs_cur"][i], traj["obs_next"][i])))
        new_y.append(traj["action"][i])

print(len(new_x))
print(len(new_y))
np.save('../IGL_data/InvKin_x',new_x)
np.save('../IGL_data/InvKin_y',new_y)
