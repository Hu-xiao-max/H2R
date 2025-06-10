import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
# 设置数据路径
folder_path = "/home/hux/datasets/H2R/dataset/subject1_pose_v1_1/subject1/h1/0/cam4/hand_pose_mano"  # 替换为你的实际路径

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        # 兼容 matplotlib 3.6+
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        return np.mean(zs3d)


left_translations = []
right_translations = []
left_directions = []
right_directions = []

# 遍历所有txt文件
for filename in sorted(os.listdir(folder_path)):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as f:
            data = list(map(float, f.readline().strip().split()))
            if len(data) == 124:
                # 提取左手
                left_translation = np.array(data[1:4])
                left_pose_root = np.array(data[4:7])  # 第一个关节旋转
                left_translations.append(left_translation)
                left_directions.append(left_pose_root / np.linalg.norm(left_pose_root + 1e-8))

                # 提取右手
                right_translation = np.array(data[63:66])
                right_pose_root = np.array(data[66:69])
                right_translations.append(right_translation)
                right_directions.append(right_pose_root / np.linalg.norm(right_pose_root + 1e-8))

# 转换为numpy数组
left_translations = np.array(left_translations)
right_translations = np.array(right_translations)
left_directions = np.array(left_directions)
right_directions = np.array(right_directions)

# 可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Hand Root Translation & Pose Direction")

arrow_scale = 0.02  # 控制箭头长度

# 左手
ax.plot(left_translations[:,0], left_translations[:,1], left_translations[:,2], c='gray', label="Left Hand")
for pos, dir_vec in zip(left_translations, left_directions):
    ax.quiver(pos[0], pos[1], pos[2], arrow_scale, 0, 0, color='r', alpha=0.6)  # x 轴方向 红
    ax.quiver(pos[0], pos[1], pos[2], 0, arrow_scale, 0, color='g', alpha=0.6)  # y 轴方向 绿
    ax.quiver(pos[0], pos[1], pos[2], 0, 0, arrow_scale, color='b', alpha=0.6)  # z 轴方向 蓝

# 右手
ax.plot(right_translations[:,0], right_translations[:,1], right_translations[:,2], c='black', label="Right Hand")
for pos, dir_vec in zip(right_translations, right_directions):
    ax.quiver(pos[0], pos[1], pos[2], arrow_scale, 0, 0, color='r', alpha=0.6)
    ax.quiver(pos[0], pos[1], pos[2], 0, arrow_scale, 0, color='g', alpha=0.6)
    ax.quiver(pos[0], pos[1], pos[2], 0, 0, arrow_scale, color='b', alpha=0.6)



ax.legend()
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.tight_layout()
plt.show()
