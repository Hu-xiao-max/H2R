import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取 txt 文件
file_path = "/home/hux/datasets/h2odataset/dataset/subject4_pose_v1_1/subject4/h1/0/cam4/obj_pose/000000.txt"  # 替换为实际路径
with open(file_path, 'r') as f:
    line = f.readline().strip()

# 将字符串拆分成浮点数列表
data = list(map(float, line.split()))

# 提取21个3D点（排除第一个类别编号）
points = np.array(data[1:]).reshape(21, 3)

# 分别提取x, y, z坐标
x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

# 创建3D可视化
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='b', s=50)

# 可选：标注点编号
for i, (xi, yi, zi) in enumerate(points):
    ax.text(xi, yi, zi, str(i), fontsize=8)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Visualization of 21 Points')
plt.show()
