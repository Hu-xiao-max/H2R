import os
import numpy as np
import matplotlib.pyplot as plt

# 提取左右手第一个点轨迹并可视化

# 设置目录路径
directory = "/home/hux/datasets/H2R/dataset/subject4_pose_v1_1/subject4/h1/0/cam4/hand_pose"  # <-- 替换成包含所有 .txt 文件的文件夹路径

left_hand_traj = []
right_hand_traj = []

# 遍历所有txt文件
for filename in sorted(os.listdir(directory)):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r') as f:
            line = f.readline().strip()
            numbers = [float(x) for x in line.split()]
            if len(numbers) != 128:
                print(f"跳过文件 {filename}：长度错误（{len(numbers)}）")
                continue
            
            # 读取左手第一个点（第2~4个数）
            left_point = numbers[1:4]  # 跳过第一个标志位
            right_point = numbers[65:68]  # 跳过第二个标志位
            
            left_hand_traj.append(left_point)
            right_hand_traj.append(right_point)

# 转换为 numpy 数组
left_hand_traj = np.array(left_hand_traj)
right_hand_traj = np.array(right_hand_traj)

# 可视化轨迹
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(left_hand_traj[:, 0], left_hand_traj[:, 1], left_hand_traj[:, 2], label="Left Hand", marker='o')
ax.plot(right_hand_traj[:, 0], right_hand_traj[:, 1], right_hand_traj[:, 2], label="Right Hand", marker='^')

ax.set_title("Trajectory of First Keypoint (Left and Right Hand)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.show()
