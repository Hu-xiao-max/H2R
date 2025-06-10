import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.gridspec as gridspec

# ---------- 设置路径 ----------
hand_folder_path = "/home/hux/datasets/H2R/test_dataset/hand_pose"
object_folder_path = "/home/hux/datasets/H2R/test_dataset/obj_pose"

# ---------- 读取数据 ----------
left_hand_traj = []
right_hand_traj = []
object_traj = []

hand_filenames = sorted([f for f in os.listdir(hand_folder_path) if f.endswith(".txt")])
object_filenames = sorted([f for f in os.listdir(object_folder_path) if f.endswith(".txt")])
n_frames = min(len(hand_filenames), len(object_filenames))

for hf, of in zip(hand_filenames, object_filenames):
    with open(os.path.join(hand_folder_path, hf), 'r') as f:
        data = list(map(float, f.readline().strip().split()))
        if len(data) == 128:
            left_hand_traj.append(data[1:4])
            right_hand_traj.append(data[65:68])
    
    with open(os.path.join(object_folder_path, of), 'r') as f:
        data = list(map(float, f.readline().strip().split()))
        if len(data) >= 4:
            object_traj.append(data[1:4])

left_hand_traj = np.array(left_hand_traj)
right_hand_traj = np.array(right_hand_traj)
object_traj = np.array(object_traj)

# ---------- Cube 绘制函数 ----------
def create_cube(center, size=0.02, color='gray'):
    """Return Poly3DCollection of a cube"""
    x, y, z = center
    d = size / 2
    # 定义立方体8个顶点
    vertices = [
        [x - d, y - d, z - d],
        [x + d, y - d, z - d],
        [x + d, y + d, z - d],
        [x - d, y + d, z - d],
        [x - d, y - d, z + d],
        [x + d, y - d, z + d],
        [x + d, y + d, z + d],
        [x - d, y + d, z + d],
    ]
    # 定义6个面
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[4], vertices[7], vertices[3], vertices[0]],
    ]
    cube = Poly3DCollection(faces, alpha=0.8)
    cube.set_facecolor(color)
    return cube

# ---------- 图像初始化 ----------
fig = plt.figure(figsize=(14, 12))

# 创建网格：1行2列
gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

# 左边大图：3D轨迹图
ax_3d = fig.add_subplot(gs[0, 0], projection='3d')

# 右边三个subplot：纵向叠放
gs_right = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0, 1], hspace=0.3)

ax_x = fig.add_subplot(gs_right[0])
ax_y = fig.add_subplot(gs_right[1])
ax_z = fig.add_subplot(gs_right[2])

# 准备画线
# 初始化折线图：空数据 g- 代表绿色的线
line_x, = ax_x.plot([], [], 'g-', label='X')
line_y, = ax_y.plot([], [], 'g-', label='Y')
line_z, = ax_z.plot([], [], 'g-', label='Z')

# 当前帧指示线 subplot中的点 go代表绿色 ro代表红色 bo代表蓝色
point_x, = ax_x.plot([], [], 'go')
point_y, = ax_y.plot([], [], 'go')
point_z, = ax_z.plot([], [], 'go')

# 设置标题与坐标轴
ax_x.set_ylabel("X")
ax_y.set_ylabel("Y")
ax_z.set_ylabel("Z")
ax_z.set_xlabel("Time (frame)")

ax_x.set_xlim(0, len(object_traj))
ax_y.set_xlim(0, len(object_traj))
ax_z.set_xlim(0, len(object_traj))

ax_x.set_title("Object X over Time")
ax_y.set_title("Object Y over Time")
ax_z.set_title("Object Z over Time")


# 静态轨迹线
left_line, = ax_3d.plot([], [], [], 'r--', label="Left Hand")
right_line, = ax_3d.plot([], [], [], 'b--', label="Right Hand")
object_line, = ax_3d.plot([], [], [], 'g--', label="Object")

# 当前点
left_scatter, = ax_3d.plot([], [], [], 'ro')
right_scatter, = ax_3d.plot([], [], [], 'bo')
object_scatter, = ax_3d.plot([], [], [], 'go')

# 初始化 cubes（固定位置）
# Cube位置：3个object + 3个左手 + 3个右手
cube_positions = [
    (-0.3, 0.3, 0.3), (-0.25, 0.3, 0.3), (-0.2, 0.3, 0.3),      # object xyz
    (-0.3, 0.25, 0.3), (-0.25, 0.25, 0.3), (-0.2, 0.25, 0.3),    # left hand xyz
    (-0.3, 0.2, 0.3), (-0.25, 0.2, 0.3), (-0.2, 0.2, 0.3),       # right hand xyz
]

cube_artists = [create_cube(pos) for pos in cube_positions]
for cube in cube_artists:
    ax_3d.add_collection3d(cube)

# ---------- 设置轴范围 ----------
ax_3d.set_xlim(np.min(object_traj[:, 0]) - 0.2, np.max(object_traj[:, 0]) + 0.2)
ax_3d.set_ylim(np.min(object_traj[:, 1]) - 0.2, np.max(object_traj[:, 1]) + 0.2)
ax_3d.set_zlim(np.min(object_traj[:, 2]) - 0.2, np.max(object_traj[:, 2]) + 0.2)
ax_3d.set_xlabel("X")
ax_3d.set_ylabel("Y")
ax_3d.set_zlabel("Z")
ax_3d.set_title("Animated Trajectory with XYZ Change Indicators")
ax_3d.legend()

# ---------- 更新函数 ----------
def update(frame):
    l, r, o = left_hand_traj[frame], right_hand_traj[frame], object_traj[frame]

    # 更新点和轨迹
    left_scatter.set_data([l[0]], [l[1]])
    left_scatter.set_3d_properties([l[2]])
    right_scatter.set_data([r[0]], [r[1]])
    right_scatter.set_3d_properties([r[2]])
    object_scatter.set_data([o[0]], [o[1]])
    object_scatter.set_3d_properties([o[2]])

    left_line.set_data(left_hand_traj[:frame+1, 0], left_hand_traj[:frame+1, 1])
    left_line.set_3d_properties(left_hand_traj[:frame+1, 2])
    right_line.set_data(right_hand_traj[:frame+1, 0], right_hand_traj[:frame+1, 1])
    right_line.set_3d_properties(right_hand_traj[:frame+1, 2])
    object_line.set_data(object_traj[:frame+1, 0], object_traj[:frame+1, 1])
    object_line.set_3d_properties(object_traj[:frame+1, 2])

    # ---- 判断亮灯 ----
    # ---- 判断亮灯逻辑 ----
    if frame == 0:
        obj_diffs = [0.0, 0.0, 0.0]
        left_diffs = [0.0, 0.0, 0.0]
        right_diffs = [0.0, 0.0, 0.0]
    else:
        obj_diffs = np.abs(object_traj[frame] - object_traj[frame - 1])
        left_diffs = np.abs(left_hand_traj[frame] - left_hand_traj[frame - 1])
        right_diffs = np.abs(right_hand_traj[frame] - right_hand_traj[frame - 1])

    # 设置 cube 颜色
    color_map = lambda diff, base_color: base_color if diff > 0.001 else 'gray'

    cube_colors = [
        color_map(obj_diffs[0], 'green'),   # object x
        color_map(obj_diffs[1], 'green'), # object y
        color_map(obj_diffs[2], 'green'),  # object z

        color_map(left_diffs[0], 'red'),   # left hand x
        color_map(left_diffs[1], 'red'), # left hand y
        color_map(left_diffs[2], 'red'),  # left hand z

        color_map(right_diffs[0], 'blue'),   # right hand x
        color_map(right_diffs[1], 'blue'), # right hand y
        color_map(right_diffs[2], 'blue'),  # right hand z
    ]

    # 删除旧方块
    for cube in cube_artists:
        cube.remove()

    # 重新添加新方块
    new_cubes = []
    for i in range(len(cube_positions)):
        cube = create_cube(cube_positions[i], color=cube_colors[i])
        ax_3d.add_collection3d(cube)
        new_cubes.append(cube)

    cube_artists[:] = new_cubes


    # 更新轨迹曲线（截断到当前帧）
    line_x.set_data(np.arange(frame + 1), object_traj[:frame + 1, 0])
    line_y.set_data(np.arange(frame + 1), object_traj[:frame + 1, 1])
    line_z.set_data(np.arange(frame + 1), object_traj[:frame + 1, 2])

    # 更新当前帧在折线图中的点
    point_x.set_data([frame], [object_traj[frame, 0]])
    point_y.set_data([frame], [object_traj[frame, 1]])
    point_z.set_data([frame], [object_traj[frame, 2]])

    # 更新 y 轴自动范围（可选）
    ax_x.set_ylim(np.min(object_traj[:, 0]) - 0.05, np.max(object_traj[:, 0]) + 0.05)
    ax_y.set_ylim(np.min(object_traj[:, 1]) - 0.05, np.max(object_traj[:, 1]) + 0.05)
    ax_z.set_ylim(np.min(object_traj[:, 2]) - 0.05, np.max(object_traj[:, 2]) + 0.05)



    return (left_scatter, right_scatter, object_scatter, left_line, right_line, object_line, 
            *cube_artists, 
            line_x, point_x,
            line_y, point_y,
            line_z, point_z)


# ---------- 动画 ----------
ani = FuncAnimation(fig, update, frames=n_frames, interval=200, blit=False)
plt.tight_layout()
plt.show()
