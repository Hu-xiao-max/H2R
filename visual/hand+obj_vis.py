import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置左右手数据路径
hand_folder_path = "/home/hux/datasets/H2R/dataset/subject4_pose_v1_1/subject4/h1/0/cam4/hand_pose"  # 替换为你的左右手txt目录
# 设置物体轨迹数据路径
object_folder_path = "/home/hux/datasets/H2R/dataset/subject4_pose_v1_1/subject4/h1/0/cam4/obj_pose"

left_hand_traj = []
right_hand_traj = []
object_traj = []

# 遍历左右手txt文件（假设与obj_pose文件数量顺序对齐）
hand_filenames = sorted([f for f in os.listdir(hand_folder_path) if f.endswith(".txt")])
object_filenames = sorted([f for f in os.listdir(object_folder_path) if f.endswith(".txt")])

for hf, of in zip(hand_filenames, object_filenames):
    # ---- 读取左右手第一个点 ----
    hand_path = os.path.join(hand_folder_path, hf)
    with open(hand_path, 'r') as f:
        line = f.readline().strip()
        data = [float(x) for x in line.split()]
        if len(data) == 128:
            left_point = data[1:4]     # 第一个左手点
            right_point = data[65:68]  # 第一个右手点
            left_hand_traj.append(left_point)
            right_hand_traj.append(right_point)

    # ---- 读取物体中心点 ----
    obj_path = os.path.join(object_folder_path, of)
    with open(obj_path, 'r') as f:
        line = f.readline().strip()
        data = [float(x) for x in line.split()]
        if len(data) >= 4:
            object_traj.append(data[1:4])  # 去掉类别标签，取中心点坐标

# 转换为 numpy 数组
left_hand_traj = np.array(left_hand_traj)
right_hand_traj = np.array(right_hand_traj)
object_traj = np.array(object_traj)

# --- 可视化 ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 左手轨迹
ax.plot(left_hand_traj[:, 0], left_hand_traj[:, 1], left_hand_traj[:, 2],
        marker='o', label="Left Hand", color='red')

# 右手轨迹
ax.plot(right_hand_traj[:, 0], right_hand_traj[:, 1], right_hand_traj[:, 2],
        marker='^', label="Right Hand", color='blue')

# 物体轨迹
ax.plot(object_traj[:, 0], object_traj[:, 1], object_traj[:, 2],
        marker='s', label="Object Center", color='green')
ax.scatter(object_traj[0, 0], object_traj[0, 1], object_traj[0, 2], c='g', s=60, label='Start')
ax.scatter(object_traj[-1, 0], object_traj[-1, 1], object_traj[-1, 2], c='r', s=60, label='End')

# 坐标轴与图例
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Left/Right Hand and Object Trajectories")
ax.legend()
plt.tight_layout()
plt.show()
