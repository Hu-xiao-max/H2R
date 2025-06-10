import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg




# ---------- 设置路径 ----------
hand_folder_path   = "/home/hux/datasets/H2R/test_dataset/hand_pose"
object_folder_path = "/home/hux/datasets/H2R/test_dataset/obj_pose"
image_folder       = "/home/hux/datasets/H2R/test_dataset/rgb256"

# 手和物体轨迹
left_hand_traj, right_hand_traj, object_traj = [], [], []
hands = sorted(f for f in os.listdir(hand_folder_path)   if f.endswith(".txt"))
objs  = sorted(f for f in os.listdir(object_folder_path) if f.endswith(".txt"))
n_frames = min(len(hands), len(objs))

for hf, of in zip(hands, objs):
    # 左右手第1点
    d = np.loadtxt(os.path.join(hand_folder_path, hf))
    if d.size == 128:
        left_hand_traj.append(d[1:4])
        right_hand_traj.append(d[65:68])
    # 物体中心
    d = np.loadtxt(os.path.join(object_folder_path, of))
    if d.size >= 4:
        object_traj.append(d[1:4])

left_hand_traj = np.array(left_hand_traj)
right_hand_traj = np.array(right_hand_traj)
object_traj    = np.array(object_traj)

# 图片序列
imgs = []
img_files = sorted(f for f in os.listdir(image_folder) if f.lower().endswith(".jpg"))
for fn in img_files:
    imgs.append(mpimg.imread(os.path.join(image_folder, fn)))

# ========== 2. 帮助函数：生成立方体 ==========
def create_cube(center, size=0.02, color='gray'):
    x,y,z = center; d = size/2
    verts = [
        [x-d,y-d,z-d],[x+d,y-d,z-d],[x+d,y+d,z-d],[x-d,y+d,z-d],
        [x-d,y-d,z+d],[x+d,y-d,z+d],[x+d,y+d,z+d],[x-d,y+d,z+d],
    ]
    faces = [
        [verts[0],verts[1],verts[2],verts[3]],
        [verts[4],verts[5],verts[6],verts[7]],
        [verts[0],verts[1],verts[5],verts[4]],
        [verts[2],verts[3],verts[7],verts[6]],
        [verts[1],verts[2],verts[6],verts[5]],
        [verts[4],verts[7],verts[3],verts[0]],
    ]
    pc = Poly3DCollection(faces, alpha=0.8)
    pc.set_facecolor(color)
    return pc

# ========== 3. 布局：1x2 + 右侧4行子图 ==========
fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

# 左侧 3D 大图
ax_3d = fig.add_subplot(gs[0,0], projection='3d')

# 右侧纵向 4 个区块
# 右侧 4 行子图，设置 height_ratios：前三行各占 1 单位，高度缩小；底行占 3 单位，高度放大
gs_r = gridspec.GridSpecFromSubplotSpec(
    4, 1,
    subplot_spec=gs[0, 1],
    hspace=0.4,
    height_ratios=[1, 1, 1, 3]    # <- 这里
)

ax_x   = fig.add_subplot(gs_r[0])
ax_y   = fig.add_subplot(gs_r[1])
ax_z   = fig.add_subplot(gs_r[2])
ax_img = fig.add_subplot(gs_r[3])
ax_img.axis('off')

# ========== 4. 初始化 Artists ==========
# ——— 3D 轨迹 & 散点
left_line,  = ax_3d.plot([],[],[], 'r--', label="Left Hand")
right_line, = ax_3d.plot([],[],[], 'b--', label="Right Hand")
obj_line,   = ax_3d.plot([],[],[], 'g--', label="Object")
left_sc,    = ax_3d.plot([],[],[], 'ro')
right_sc,   = ax_3d.plot([],[],[], 'bo')
obj_sc,     = ax_3d.plot([],[],[], 'go')

# ——— 9 个 cube 指示灯
cube_positions = [
    (-0.3,0.3,0.3),(-0.25,0.3,0.3),(-0.2,0.3,0.3),   # obj XYZ
    (-0.3,0.25,0.3),(-0.25,0.25,0.3),(-0.2,0.25,0.3),# left XYZ
    (-0.3,0.2,0.3),(-0.25,0.2,0.3),(-0.2,0.2,0.3),   # right XYZ
]
cube_artists = []
for pos in cube_positions:
    c = create_cube(pos)
    ax_3d.add_collection3d(c)
    cube_artists.append(c)

# ——— 三条空折线 + 当前点
line_x, = ax_x.plot([], [], 'r-')
pt_x,   = ax_x.plot([], [], 'ro')
line_y, = ax_y.plot([], [], 'g-')
pt_y,   = ax_y.plot([], [], 'go')
line_z, = ax_z.plot([], [], 'b-')
pt_z,   = ax_z.plot([], [], 'bo')

# ——— 图片模版
im = ax_img.imshow(imgs[0])

# ========== 5. 坐标轴 & 图例 设置 ==========
# 3D 轴范围、标签
ax_3d.set_xlim(object_traj[:,0].min()-0.2, object_traj[:,0].max()+0.2)
ax_3d.set_ylim(object_traj[:,1].min()-0.2, object_traj[:,1].max()+0.2)
ax_3d.set_zlim(object_traj[:,2].min()-0.2, object_traj[:,2].max()+0.2)
ax_3d.set_xlabel("X"); ax_3d.set_ylabel("Y"); ax_3d.set_zlabel("Z")
ax_3d.set_title("3D Hand+Object Trajectories")
ax_3d.legend(loc='upper left')

# 折线图设置
for ax, lbl in zip((ax_x,ax_y,ax_z), ("Object X","Object Y","Object Z")):
    ax.set_xlim(0, n_frames)
    ax.set_ylabel(lbl)
ax_z.set_xlabel("Frame")

# ========== 6. 更新函数 ==========
def update(frame):
    # 6.1 3D 轨迹 + 散点
    l, r, o = left_hand_traj[frame], right_hand_traj[frame], object_traj[frame]
    left_sc.set_data([l[0]],[l[1]]);  left_sc.set_3d_properties([l[2]])
    right_sc.set_data([r[0]],[r[1]]); right_sc.set_3d_properties([r[2]])
    obj_sc.set_data([o[0]],[o[1]]);   obj_sc.set_3d_properties([o[2]])
    left_line.set_data(left_hand_traj[:frame+1,0], left_hand_traj[:frame+1,1])
    left_line.set_3d_properties(left_hand_traj[:frame+1,2])
    right_line.set_data(right_hand_traj[:frame+1,0], right_hand_traj[:frame+1,1])
    right_line.set_3d_properties(right_hand_traj[:frame+1,2])
    obj_line.set_data(object_traj[:frame+1,0], object_traj[:frame+1,1])
    obj_line.set_3d_properties(object_traj[:frame+1,2])

    # 6.2 计算变化并更新 cube 颜色
    if frame == 0:
        od, ld, rd = [0, 0, 0], [0, 0, 0], [0, 0, 0]
    else:
        od = np.abs(object_traj[frame] - object_traj[frame - 1]).tolist()
        ld = np.abs(left_hand_traj[frame] - left_hand_traj[frame - 1]).tolist()
        rd = np.abs(right_hand_traj[frame] - right_hand_traj[frame - 1]).tolist()

    diffs = od + ld + rd
    cols  = [c if diffs[i]>0.001 else 'gray'
             for i,c in enumerate(
                ['red','green','blue']*3
             )]
    # remove old & add new
    for c in cube_artists: c.remove()
    for i,pos in enumerate(cube_positions):
        cb = create_cube(pos, color=cols[i])
        ax_3d.add_collection3d(cb)
        cube_artists[i] = cb

    # 6.3 更新折线图
    xs = np.arange(frame+1)
    line_x.set_data(xs, object_traj[:frame+1,0]); pt_x.set_data([frame],[object_traj[frame,0]])
    line_y.set_data(xs, object_traj[:frame+1,1]); pt_y.set_data([frame],[object_traj[frame,1]])
    line_z.set_data(xs, object_traj[:frame+1,2]); pt_z.set_data([frame],[object_traj[frame,2]])
    for ax,d in zip((ax_x,ax_y,ax_z),(object_traj[:,0],object_traj[:,1],object_traj[:,2])):
        mn, mx = d.min()-0.05, d.max()+0.05
        ax.set_ylim(mn, mx)

    # 6.4 更新图像
    im.set_data(imgs[frame])

    return (
        left_sc, right_sc, obj_sc,
        left_line, right_line, obj_line,
        *cube_artists,
        line_x, pt_x, line_y, pt_y, line_z, pt_z,
        im
    )

# ========== 7. 生成动画 ==========
ani = FuncAnimation(fig, update, frames=n_frames, interval=200, blit=False)
plt.show()
