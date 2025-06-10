import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# 路径设置
hand_folder_path = "/home/hux/datasets/H2R/dataset/subject4_pose_v1_1/subject4/h1/0/cam4/hand_pose"  # 替换成你的路径
object_folder_path = "/home/hux/datasets/H2R/dataset/subject4_pose_v1_1/subject4/h1/0/cam4/obj_pose"

# 数据加载
left_hand_traj = []
right_hand_traj = []
object_traj = []

# 获取文件列表（假设按文件名顺序对齐）
hand_filenames = sorted([f for f in os.listdir(hand_folder_path) if f.endswith(".txt")])
object_filenames = sorted([f for f in os.listdir(object_folder_path) if f.endswith(".txt")])
n_frames = min(len(hand_filenames), len(object_filenames))

for hf, of in zip(hand_filenames, object_filenames):
    # 读取手部
    with open(os.path.join(hand_folder_path, hf), 'r') as f:
        data = list(map(float, f.readline().strip().split()))
        if len(data) == 128:
            left_hand_traj.append(data[1:4])
            right_hand_traj.append(data[65:68])
    
    # 读取物体中心
    with open(os.path.join(object_folder_path, of), 'r') as f:
        data = list(map(float, f.readline().strip().split()))
        if len(data) >= 4:
            object_traj.append(data[1:4])

# 转为 numpy 数组
left_hand_traj = np.array(left_hand_traj)
right_hand_traj = np.array(right_hand_traj)
object_traj = np.array(object_traj)

# 初始化图像
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 预设图形元素
left_scatter, = ax.plot([], [], [], 'ro', label='Left Hand')
right_scatter, = ax.plot([], [], [], 'bo', label='Right Hand')
object_scatter, = ax.plot([], [], [], 'go', label='Object')

# 可选：添加历史轨迹线
left_line, = ax.plot([], [], [], 'r--', linewidth=1)
right_line, = ax.plot([], [], [], 'b--', linewidth=1)
object_line, = ax.plot([], [], [], 'g--', linewidth=1)

# 设置图像基本参数
ax.set_xlim(np.min(object_traj[:, 0]) - 0.2, np.max(object_traj[:, 0]) + 0.2)
ax.set_ylim(np.min(object_traj[:, 1]) - 0.2, np.max(object_traj[:, 1]) + 0.2)
ax.set_zlim(np.min(object_traj[:, 2]) - 0.2, np.max(object_traj[:, 2]) + 0.2)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Animated Hand + Object Trajectory")
ax.legend()

# 更新函数
def update(frame):
    # 当前帧数据
    l = left_hand_traj[frame]
    r = right_hand_traj[frame]
    o = object_traj[frame]

    # 更新散点位置
    left_scatter.set_data([l[0]], [l[1]])
    left_scatter.set_3d_properties([l[2]])
    
    right_scatter.set_data([r[0]], [r[1]])
    right_scatter.set_3d_properties([r[2]])

    object_scatter.set_data([o[0]], [o[1]])
    object_scatter.set_3d_properties([o[2]])

    # 可选：历史轨迹连线
    left_line.set_data(left_hand_traj[:frame+1, 0], left_hand_traj[:frame+1, 1])
    left_line.set_3d_properties(left_hand_traj[:frame+1, 2])

    right_line.set_data(right_hand_traj[:frame+1, 0], right_hand_traj[:frame+1, 1])
    right_line.set_3d_properties(right_hand_traj[:frame+1, 2])

    object_line.set_data(object_traj[:frame+1, 0], object_traj[:frame+1, 1])
    object_line.set_3d_properties(object_traj[:frame+1, 2])

    return left_scatter, right_scatter, object_scatter, left_line, right_line, object_line

# 动画创建
ani = FuncAnimation(fig, update, frames=n_frames, interval=200, blit=False)

plt.tight_layout()
plt.show()
