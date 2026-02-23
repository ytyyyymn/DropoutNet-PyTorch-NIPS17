import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from model import DropoutNet
from dataset import load_data, RecSysDataset

# --- 超参数配置 ---
BASE_DIR = './data/recsys2017.pub'
LATENT_DIM = 200
BATCH_SIZE = 4096
EPOCHS = 30
LR = 0.005                     # 【修改】回调到 0.005，保持稳定
DROPOUT_RATE = 0.5
HIDDEN_DIMS = [800, 800, 400]


def plot_loss_curve(step_losses, epoch_losses):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(step_losses, label='Step Loss', color='tab:blue', alpha=0.6)
    plt.title('Training Loss (Per 50 Steps)')
    plt.xlabel('Steps (x50)')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', label='Epoch Avg Loss', color='red')
    plt.title('Average Training Loss (Per Epoch)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.legend()

    save_path = 'training_loss_curve.png'
    plt.savefig(save_path)
    print(f"\n📊 Loss 曲线已保存为: {save_path}")
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    # 1. 加载数据
    interactions, u_pref, v_pref, u_content, v_content = load_data(BASE_DIR, LATENT_DIM)

    num_users = u_pref.size(0)
    num_items = v_pref.size(0)

    # 2. 【核心新增】预计算 Section 4.3 的 Inference Transform 均值向量
    print("=== 正在预计算 Inference Transform 均值向量 (Section 4.3) ===")
    row = interactions[:, 0]
    col = interactions[:, 1]
    data = np.ones(len(interactions))
    adj_matrix = sp.csr_matrix((data, (row, col)), shape=(num_users, num_items))

    # 计算 u_pref_mean (公式 4)
    user_degrees = adj_matrix.sum(axis=1).A1
    user_degrees[user_degrees == 0] = 1
    u_pref_mean = torch.FloatTensor(adj_matrix.dot(v_pref.numpy()) / user_degrees[:, None])

    # 计算 v_pref_mean
    item_degrees = adj_matrix.sum(axis=0).A1
    item_degrees[item_degrees == 0] = 1
    v_pref_mean = torch.FloatTensor(adj_matrix.T.dot(u_pref.numpy()) / item_degrees[:, None])

    # 挂载到 GPU
    u_pref_mean_gpu = u_pref_mean.to(device)
    v_pref_mean_gpu = v_pref_mean.to(device)
    print("=== 均值向量计算完成！ ===")

    # 3. 初始化 Dataset 和 Dataloader
    dataset = RecSysDataset(interactions, num_users, num_items)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # 4. 初始化模型
    u_content_dim = u_content.shape[1]
    v_content_dim = v_content.shape[1]

    model = DropoutNet(
        u_latent_dim=LATENT_DIM,
        u_content_dim=u_content_dim,
        v_latent_dim=LATENT_DIM,
        v_content_dim=v_content_dim,
        hidden_dims=HIDDEN_DIMS,
        dropout_rate=DROPOUT_RATE
    ).to(device)

    # 5. 优化器
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    criterion = nn.MSELoss()

    step_losses = []
    epoch_losses = []

    print(f"=== 开始极速训练 DropoutNet (Total Epochs: {EPOCHS}) ===")
    model.train()

    u_pref_gpu = u_pref.to(device)
    v_pref_gpu = v_pref.to(device)

    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}", unit="batch", leave=True)

        for batch_idx, (batch_uids, batch_iids) in enumerate(pbar):

            u_p = u_pref_gpu[batch_uids]
            v_p = v_pref_gpu[batch_iids]

            # 并行计算 Ground Truth
            target = (u_p * v_p).sum(dim=1)

            # 批量读取内容特征并转 Tensor
            u_c_sparse = u_content[batch_uids.numpy()]
            v_c_sparse = v_content[batch_iids.numpy()]
            u_c = torch.from_numpy(u_c_sparse.toarray().astype(np.float32)).to(device)
            v_c = torch.from_numpy(v_c_sparse.toarray().astype(np.float32)).to(device)

            optimizer.zero_grad()

            # --- 交替进行 Input Dropout 与 Inference Transform ---
            mask_u = torch.bernoulli(torch.full((u_p.size(0), 1), 1 - DROPOUT_RATE)).to(device)
            mask_v = torch.bernoulli(torch.full((v_p.size(0), 1), 1 - DROPOUT_RATE)).to(device)

            if batch_idx % 2 == 0:
                # 策略 A: 纯冷启动模拟 (置零)
                u_p_input = u_p * mask_u
                v_p_input = v_p * mask_v
            else:
                # 策略 B: 灰度过渡模拟 (使用交互均值替换)
                u_p_input = torch.where(mask_u == 1, u_p, u_pref_mean_gpu[batch_uids])
                v_p_input = torch.where(mask_v == 1, v_p, v_pref_mean_gpu[batch_iids])

            # 前向传播 (关闭模型内部的 dropout，因为我们在外部手动做好了)
            scores, _, _ = model(u_p_input, u_c, v_p_input, v_c, apply_dropout=False)

            loss = criterion(scores, target)
            loss.backward()

            # 【核心急救代码】：梯度裁剪，防止 NaN！
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            current_loss = loss.item()
            total_loss += current_loss
            pbar.set_postfix({'loss': f'{current_loss:.4f}'})

            if batch_idx % 50 == 0:
                step_losses.append(current_loss)

        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)

    torch.save(model.state_dict(), "dropoutnet_recsys_model.pth")
    print("✅ 模型已成功保存！")
    plot_loss_curve(step_losses, epoch_losses)


if __name__ == "__main__":
    main()