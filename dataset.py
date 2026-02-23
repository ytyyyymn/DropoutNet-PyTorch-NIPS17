import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from sklearn.datasets import load_svmlight_file


# 请用这段代码替换 dataset.py 中的 RecSysDataset 类
# (load_data 函数保持不变)

class RecSysDataset(Dataset):
    def __init__(self, interactions, num_users, num_items):
        """
        极简版 Dataset：只负责生成正负样本的 ID对，极大地加速 DataLoader
        """
        self.interactions = interactions
        self.num_users = num_users
        self.num_items = num_items

    def __len__(self):
        return len(self.interactions) * 2  # 一半正样本，一半负样本

    def __getitem__(self, idx):
        if idx % 2 == 0:
            uid, iid = self.interactions[idx // 2]
        else:
            uid = np.random.randint(0, self.num_users)
            iid = np.random.randint(0, self.num_items)

        return uid, iid


def load_data(base_dir, latent_dim=200):
    """
    加载 RecSys 2017 数据集
    """
    print(f"正在从 {base_dir} 加载数据...")

    # --- 1. 定义路径 ---
    u_path = os.path.join(base_dir, 'eval/trained/warm/U.csv.bin')
    v_path = os.path.join(base_dir, 'eval/trained/warm/V.csv.bin')
    train_csv = os.path.join(base_dir, 'eval/warm/train.csv')
    u_feat_path = os.path.join(base_dir, 'eval/user_features_0based.txt')
    i_feat_path = os.path.join(base_dir, 'eval/item_features_0based.txt')

    # --- 2. 加载 WMF 潜向量 ---
    print("加载 WMF 向量...")
    u_pref_np = np.fromfile(u_path, dtype=np.float32)
    v_pref_np = np.fromfile(v_path, dtype=np.float32)

    num_users = int(u_pref_np.shape[0] / latent_dim)
    num_items = int(v_pref_np.shape[0] / latent_dim)

    u_pref = torch.FloatTensor(u_pref_np.reshape(num_users, latent_dim))
    v_pref = torch.FloatTensor(v_pref_np.reshape(num_items, latent_dim))

    # --- 3. 加载内容特征 (保持稀疏!) ---
    print("加载内容特征 (保持稀疏格式以节省内存)...")
    # load_svmlight_file 返回的是 scipy.sparse.csr_matrix
    # 我们不再在这里调用 .toarray()，而是直接把这个压缩对象传出去
    u_content = load_svmlight_file(u_feat_path, n_features=None)[0]
    v_content = load_svmlight_file(i_feat_path, n_features=None)[0]

    # 转换为 float32 格式的稀疏矩阵 (防止 double 精度占用过多内存)
    u_content = u_content.astype(np.float32)
    v_content = v_content.astype(np.float32)

    # --- 4. 加载交互数据 ---
    print("加载交互记录...")
    df = pd.read_csv(train_csv, header=None, usecols=[0, 1])
    interactions = df.values.astype(np.int64)

    print("数据加载完成！")
    print(f"用户数: {num_users}, 物品数: {num_items}")
    # 打印一下稀疏矩阵的维度
    print(f"用户特征维度: {u_content.shape}, 物品特征维度: {v_content.shape}")
    print(f"训练样本数: {len(interactions)}")

    return interactions, u_pref, v_pref, u_content, v_content