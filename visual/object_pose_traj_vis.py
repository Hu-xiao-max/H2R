import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取指定object pose下所有txt的第一个center点，并且组成轨迹

# 设置你的txt文件目录路径
folder_path = "/home/hux/datasets/h2odataset/dataset/subject4_pose_v1_1/subject4/h1/0/cam4/obj_pose"  # 替换为你的实际路径

trajectory = []

# 遍历目录下所有txt文件
for filename in sorted(os.listdir(folder_path)):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as f:
            line = f.readline().strip()
            data = list(map(float, line.split()))
            if len(data) >= 4:  # 至少包括1个类别编号 + 1个3D点
                x1, y1, z1 = data[1], data[2], data[3]
                trajectory.append((x1, y1, z1))

# 转换为numpy数组
trajectory = np.array(trajectory)

# 可视化轨迹
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], marker='o', label="Trajectory")
ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], c='g', s=60, label='Start')
ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], c='r', s=60, label='End')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Trajectory from First Keypoint in Each TXT File')
ax.legend()
plt.show()
