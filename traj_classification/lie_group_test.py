'''
Test For lie group
'''

import numpy as np
import os
from scipy.spatial.transform import Rotation as R

def se3_log(T):
    R_mat = T[:3, :3]
    t = T[:3, 3]
    rot = R.from_matrix(R_mat)
    omega = rot.as_rotvec()
    theta = np.linalg.norm(omega)
    if theta < 1e-8:
        V_inv = np.eye(3)
    else:
        omega_hat = omega / theta
        K = np.array([
            [0, -omega_hat[2], omega_hat[1]],
            [omega_hat[2], 0, -omega_hat[0]],
            [-omega_hat[1], omega_hat[0], 0]
        ])
        B = (1 - np.cos(theta)) / (theta**2)
        C = (theta - np.sin(theta)) / (theta**3)
        V = np.eye(3) + B * K + C * (K @ K)
        V_inv = np.linalg.inv(V)
    v = V_inv @ t
    xi = np.zeros(6)
    xi[:3] = omega
    xi[3:] = v
    return xi

# 指定你的目录
dir_path = '/home/hux/datasets/H2R/test_dataset/obj_pose_rt'  # ← 修改为你的目录名

for filename in os.listdir(dir_path):
    if filename.endswith('.txt'):
        filepath = os.path.join(dir_path, filename)
        with open(filepath, 'r') as f:
            line = f.readline().strip()
            if not line:
                continue
            data = [float(x) for x in line.split()]
            label = int(data[0])
            T_flat = data[1:17]
            T = np.array(T_flat).reshape(4, 4)
            xi = se3_log(T)
            print(f"文件: {filename}")
            print(f"物体类别: {label}")
            print("李代数向量:", np.round(xi, 6))
            print("="*40)

