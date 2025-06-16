
'''
DBSCANS For lie group clustering
'''

dir_path = '/home/hux/datasets/H2R/test_dataset/obj_pose_rt'


import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

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

# 1. 读取所有李群向量
dir_path = '/home/hux/datasets/H2R/test_dataset/obj_pose_rt'  # 替换为你的文件夹
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
            T_flat = data[1:17]
            T = np.array(T_flat).reshape(4, 4)
            xi = se3_log(T)
            all_xi.append(xi)
            file_labels.append(filename)
all_xi = np.array(all_xi)  # shape (N,6)

# K nearnest for eps select

# # all_xi 为你的 6 维向量数据
# k = 5   # 一般选min_samples或min_samples-1
# nbrs = NearestNeighbors(n_neighbors=k).fit(all_xi)
# distances, indices = nbrs.kneighbors(all_xi)
# dists = np.sort(distances[:, -1])  # 取每个点到第k近邻的距离
# plt.figure(figsize=(8,5))
# plt.plot(dists)
# plt.title('k近邻距离排序曲线，选取转折点为eps')
# plt.ylabel(f'{k}-NN 距离')
# plt.xlabel('点序号')
# plt.grid()
# plt.show()

# 2. 直接在6维空间聚类
db = DBSCAN(eps=0.04, min_samples=2)
cluster_labels = db.fit_predict(all_xi)
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = list(cluster_labels).count(-1)
print(f'共{n_clusters}个聚类, {n_noise}个噪声点')

# 3. 降维仅做可视化（这里用PCA降到2维方便画图）
pca = PCA(n_components=2)
xi_pca = pca.fit_transform(all_xi)

# 4. 可视化聚类标签（按DBSCAN标签分色）
plt.figure(figsize=(10, 7))
unique_labels = set(cluster_labels)
colors = plt.cm.get_cmap('tab10', len(unique_labels))
for label in unique_labels:
    idx = cluster_labels == label
    if label == -1:
        plt.scatter(xi_pca[idx, 0], xi_pca[idx, 1], c='gray', marker='x', s=70, label='Noise')
    else:
        plt.scatter(xi_pca[idx, 0], xi_pca[idx, 1], color=colors(label), label=f'Cluster {label}', s=60)
# for i, txt in enumerate(file_labels):
#     plt.text(xi_pca[i, 0], xi_pca[i, 1], txt, fontsize=8)
# plt.xlabel('PCA 1')
# plt.ylabel('PCA 2')
# plt.title('DBSCAN聚类结果（李群代数向量原始空间聚类，PCA降维可视化）')
plt.legend()
plt.tight_layout()
plt.show()
