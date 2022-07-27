import numpy as np

def obs2dictobs(obs):
    return dict(obs_cur_robot_pos=obs[:4],   obs_cur_obj1_pos=obs[4:7], obs_cur_obj1_quat=obs[7:11],  \
                obs_pre_robot_pos=obs[18:22],obs_pre_obj1_pos=obs[22:25], obs_pre_obj1_quat=obs[25:29], goal=obs[-3:])
def obs2BCO_state(obs):
    return np.concatenate((obs[:11], obs[18:29], obs[-3:]))

def obs2igl_state(obs,subgoal):
    return np.concatenate((obs[:11],obs[18:29],obs[-3:],subgoal))

def get_subgoal(dictobs,pre_sub_goal,task_name):
    if 'pick-place' in task_name:
        if abs(dictobs['obs_cur_robot_pos'][0] - dictobs['obs_cur_obj1_pos'][0]) < 0.007 and \
           abs(dictobs['obs_cur_robot_pos'][1] - dictobs['obs_cur_obj1_pos'][1]) < 0.035 and \
           abs(dictobs['obs_cur_robot_pos'][2] - dictobs['obs_cur_obj1_pos'][2]) < 0.045  and pre_sub_goal == 0:
            return np.array([1])
        if dictobs['obs_cur_robot_pos'][3]<0.7 and pre_sub_goal == 1:
            return np.array([2])
    if 'drawer-open' in task_name:
        if abs(dictobs['obs_cur_robot_pos'][0] - dictobs['obs_cur_obj1_pos'][0]) < 0.025 and \
           abs(dictobs['obs_cur_robot_pos'][1] - dictobs['obs_cur_obj1_pos'][1]) < 0.017 and \
           abs(dictobs['obs_cur_robot_pos'][2] - dictobs['obs_cur_obj1_pos'][2]) < 0.010  and pre_sub_goal == 0:
            return np.array([1])

    return pre_sub_goal

class logger():
    def __init__(self,file_name,iter):
        self.file_name = file_name
        self.iter = iter
        f = open("./Result/"+file_name + str(iter) + ".txt", 'w')
        f.close
    def write(self, result_list):
        f = open("./Result/"+self.file_name + str(self.iter) + ".txt", 'a')
        for re in result_list:
            f.write(str(re))
            f.write(" ")
        f.write("\n")
        f.close()


def get_subgoal_deploy(dictobs,pre_sub_goal):
    if abs(dictobs['obs_cur_robot_pos'][0] - dictobs['obs_cur_obj1_pos'][0]) < 0.018 and \
       abs(dictobs['obs_cur_robot_pos'][1] - dictobs['obs_cur_obj1_pos'][1]) < 0.045 and \
       abs(dictobs['obs_cur_robot_pos'][2] - dictobs['obs_cur_obj1_pos'][2]) < 0.053  and pre_sub_goal == 0:
        return np.array([1])
    if dictobs['obs_cur_robot_pos'][3]<0.8 and pre_sub_goal == 1:
        return np.array([2])
    return pre_sub_goal



def human_key_control(key):
    scale  = 0.6
    if "a" in key:
        a = np.array([scale,0.0,0.0,-scale])
    if "d" in key:
        a = np.array([-scale,0.0,0.0,-scale])
    if "w" in key:
        a = np.array([0.0,-scale,0.0,-scale])
    if "s" in key:
        a = np.array([0.0,scale,0.0,-scale])
    if "r" in key:
        a = np.array([0.0,0.0,scale,-scale])
    if "f" in key:
        a = np.array([0.0,0.0,-scale,-scale])

    if "m" in key:
        a = np.array([0.0, 0.0, 0.0, scale])
    if "," in key:
        a = np.array([0.0, 0.0, 0.0, -scale])

    if "." in key:
        a *= np.array([scale,scale,scale,-scale])

    return a
