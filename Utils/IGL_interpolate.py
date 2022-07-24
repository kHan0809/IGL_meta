import numpy as np
import quaternion
import math

def value_inter(v1, v2, coef):
    return v1*(1-coef) + v2*(coef)

def intpol_pos(traj1,traj2,coef):
    traj1_len = len(traj1) - 1
    traj2_len = len(traj2) - 1

    inter_traj_real_len = (traj1_len) * (1 - coef) + (traj2_len) * coef
    inter_traj_len = math.ceil(inter_traj_real_len)

    ratio1 = traj1_len / inter_traj_len;
    ratio2 = traj2_len / inter_traj_len;

    new_pos = []
    for i in range(inter_traj_len + 1):
        temp = []
        if i == 0:
            temp.extend(value_inter(traj1[0, :], traj2[0, :], coef))
        elif i == inter_traj_len:
            temp.extend(value_inter(traj1[-1, :], traj2[-1, :], coef))
        else:
            idx1 = i * ratio1
            idx2 = i * ratio2
            if (idx1 % 1.0) < 0.0001:
                new_pos1 = traj1[math.ceil(idx1),:]
            else:
                pre = math.floor(idx1)
                cur = math.ceil(idx1)
                new_pos1 = value_inter(traj1[pre,:],traj1[cur,:],(idx1%1.0))

            if (idx2 % 1.0) < 0.0001:
                new_pos2 = traj2[math.ceil(idx2), :]
            else:
                pre = math.floor(idx2)
                cur = math.ceil(idx2)
                new_pos2 = value_inter(traj2[pre,:],traj2[cur,:],(idx2%1.0))
            temp.extend(value_inter(new_pos1,new_pos2,coef))
        new_pos.append(temp)
    return np.array(new_pos)


def intpol_quat(traj1,traj2,coef):
    traj1_len = len(traj1) - 1
    traj2_len = len(traj2) - 1

    inter_traj_real_len = (traj1_len) * (1 - coef) + (traj2_len) * coef
    inter_traj_len = math.ceil(inter_traj_real_len)

    ratio1 = traj1_len / inter_traj_len;
    ratio2 = traj2_len / inter_traj_len;

    traj1_quat = np.concatenate((traj1[:, 3].reshape(-1, 1), traj1[:, :3]), axis=1)
    traj2_quat = np.concatenate((traj2[:, 3].reshape(-1, 1), traj2[:, :3]), axis=1)

    traj_q1 = quaternion.as_quat_array(traj1_quat)
    traj_q2 = quaternion.as_quat_array(traj2_quat)

    new_quat = []
    for i in range(inter_traj_len + 1):
        temp = []
        if i == 0:
            new_q = quaternion.as_float_array(quaternion.slerp_evaluate(traj_q1[0], traj_q2[0], coef))
            temp.extend(np.concatenate((new_q[1:],new_q[0].reshape(1))))
        elif i == inter_traj_len:
            new_q = quaternion.as_float_array(quaternion.slerp_evaluate(traj_q1[-1], traj_q2[-1], coef))
            temp.extend(np.concatenate((new_q[1:],new_q[0].reshape(1))))
        else:
            idx1 = i * ratio1
            idx2 = i * ratio2
            if (idx1%1.0) < 0.0001:
                new_q_1 = traj_q1[math.ceil(idx1)]
            else:
                pre = math.floor(idx1)
                cur = math.ceil(idx1)
                new_q_1 = quaternion.slerp_evaluate(traj_q1[pre], traj_q1[cur], (idx1 % 1.0))

            if (idx2 % 1.0) < 0.0001:
                new_q_2 = traj_q2[math.ceil(idx2)]
            else:
                pre = math.floor(idx2)
                cur = math.ceil(idx2)
                new_q_2 = quaternion.slerp_evaluate(traj_q2[pre], traj_q2[cur], (idx2 % 1.0))

            new_q = quaternion.as_float_array(quaternion.slerp_evaluate(new_q_1, new_q_2, coef))
            temp.extend(np.concatenate((new_q[1:],new_q[0].reshape(1))))

        new_quat.append(temp)
    return np.array(new_quat)





