import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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
dir_path = '/home/hux/datasets/H2R/test_dataset/obj_pose_rt'  # 修改为你的目录

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
all_xi = np.array(all_xi)  # (N,6)

# 2. KMeans聚类
n_clusters = 3  # 根据你的场景改聚类数
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
cluster_labels = kmeans.fit_predict(all_xi)

# 3. PCA降到2维
pca = PCA(n_components=2)
xi_pca = pca.fit_transform(all_xi)

# 4. 可视化
plt.figure(figsize=(9, 6))
for i in range(n_clusters):
    idx = cluster_labels == i
    plt.scatter(xi_pca[idx, 0], xi_pca[idx, 1], label=f'cluster {i}', s=60)
# for i, txt in enumerate(file_labels):
#     plt.text(xi_pca[i, 0], xi_pca[i, 1], txt, fontsize=8)
# plt.xlabel('PCA 1')
# plt.ylabel('PCA 2')
# plt.title('SE(3)李群代数向量 KMeans聚类可视化 (PCA降2维)')
plt.legend()
plt.tight_layout()
plt.show()
