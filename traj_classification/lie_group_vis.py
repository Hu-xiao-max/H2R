'''
Test For lie group Vis
'''


import numpy as np
import os
import matplotlib.pyplot as plt
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
dir_path = '/home/hux/datasets/H2R/test_dataset/obj_pose_rt'  # 修改为你的目录

all_xi = []
file_labels = []

for filename in sorted(os.listdir(dir_path)):
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
            all_xi.append(xi)
            file_labels.append(filename)

all_xi = np.array(all_xi)  # shape (N, 6)

# --- 可视化 ---
plt.figure(figsize=(12, 6))
comp_names = [r'$\omega_x$', r'$\omega_y$', r'$\omega_z$', r'$v_x$', r'$v_y$', r'$v_z$']
for i in range(6):
    plt.plot(all_xi[:, i], marker='o', label=comp_names[i])
plt.legend()
# plt.title('SE(3) 李群代数向量分量可视化')
# plt.xlabel('文件序号')
# plt.ylabel('分量值')
# plt.xticks(ticks=range(len(file_labels)), labels=file_labels, rotation=60)
# plt.tight_layout()
plt.show()

