import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 从txt读取
file_path = "/home/hux/datasets/h2odataset/dataset/subject4_pose_v1_1/subject4/h1/0/cam4/hand_pose/000000.txt"  # <-- 修改为你的实际文件路径

# 读取文件数据
with open(file_path, 'r') as f:
    line = f.readline().strip()

# 转为 float 数组
data = np.array([float(x) for x in line.split()])

# 检查长度是否为128
if len(data) != 128:
    raise ValueError("数据长度应为128（1 + 63 + 1 + 63），请检查txt格式")

# 提取左右手关键点
left_hand_points = data[1:64].reshape(21, 3)    # 跳过第一个标志位，取63个数
right_hand_points = data[65:128].reshape(21, 3) # 跳过第二个标志位，取63个数

# 可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制左手
ax.scatter(left_hand_points[:, 0], left_hand_points[:, 1], left_hand_points[:, 2], label='Left Hand', c='red')
ax.plot(left_hand_points[:, 0], left_hand_points[:, 1], left_hand_points[:, 2], linestyle='--', c='red')

# 绘制右手
ax.scatter(right_hand_points[:, 0], right_hand_points[:, 1], right_hand_points[:, 2], label='Right Hand', c='blue')
ax.plot(right_hand_points[:, 0], right_hand_points[:, 1], right_hand_points[:, 2], linestyle='--', c='blue')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("3D Hand Keypoints (Left and Right)")
ax.legend()

plt.show()
