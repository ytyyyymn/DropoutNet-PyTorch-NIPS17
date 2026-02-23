import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

# 复用我们已经写好的模块
from model import DropoutNet
from dataset import load_data

# --- 配置参数 ---
BASE_DIR = './data/recsys2017.pub'
MODEL_PATH = 'dropoutnet_recsys_model.pth'
LATENT_DIM = 200
HIDDEN_DIMS = [800, 800, 400]  # 必须与训练时的结构一致
NUM_SAMPLES = 2000  # 采样的物品数量 (T-SNE跑太慢，不宜过多)
SEED = 42  # 固定随机种子，保证结果可复现


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def visualize_tsne():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    # 1. 加载原始数据 (把 u_content 也接收出来)
    print("正在加载数据...")
    interactions, _, v_pref, u_content, v_content = load_data(BASE_DIR, LATENT_DIM)

    u_dim = u_content.shape[1]  # 正确获取用户的 831 维
    v_dim = v_content.shape[1]
    num_items = v_pref.size(0)

    # 2. 加载训练好的模型
    model = DropoutNet(
        u_latent_dim=LATENT_DIM,
        u_content_dim=u_dim,  # <--- 修正：传入真实的维度
        v_latent_dim=LATENT_DIM,
        v_content_dim=v_dim,
        hidden_dims=HIDDEN_DIMS
    ).to(device)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print(f"✅ 成功加载模型权重: {MODEL_PATH}")
    else:
        print(f"❌ 找不到模型文件: {MODEL_PATH}，请先完成训练。")
        return

    model.eval()  # 切换到评估模式，关闭 Dropout

    # 3. 采样物品
    # 为了对比效果，我们只挑选那些在训练集中出现过的“热物品”
    warm_item_ids = np.unique(interactions[:, 1])
    if len(warm_item_ids) > NUM_SAMPLES:
        sampled_iids = np.random.choice(warm_item_ids, NUM_SAMPLES, replace=False)
    else:
        sampled_iids = warm_item_ids
    print(f"已采样 {len(sampled_iids)} 个热物品用于可视化对比。")

    # 4. 生成两种 Embedding
    print("正在生成 Embedding...")
    with torch.no_grad():
        # 准备数据 Tensor
        v_p_batch = v_pref[sampled_iids].to(device)
        v_c_sparse_batch = v_content[sampled_iids]
        v_c_batch = torch.from_numpy(v_c_sparse_batch.toarray().astype(np.float32)).to(device)
        v_p_zero = torch.zeros_like(v_p_batch).to(device)

        # --- 生成 Warm Embeddings (理想状态: WMF + Content) ---
        input_warm = torch.cat([v_p_batch, v_c_batch], dim=1)
        # 通过物品侧网络映射到共同空间
        warm_embs = model.item_network(input_warm).cpu().numpy()

        # --- 生成 Cold Embeddings (模拟冷启动: Zero + Content) ---
        input_cold = torch.cat([v_p_zero, v_c_batch], dim=1)
        # 通过同一个网络
        cold_embs = model.item_network(input_cold).cpu().numpy()

    # 5. 运行 T-SNE 降维
    print(f"正在进行 T-SNE 降维 (从 {warm_embs.shape[1]}维 -> 2维)，请耐心等待...")
    # 将两种数据合并在一起进行降维，保证处于同一个坐标系下
    X_combined = np.vstack([warm_embs, cold_embs])
    # 创建标签：0 代表 warm, 1 代表 cold
    y_combined = np.concatenate([np.zeros(len(warm_embs)), np.ones(len(cold_embs))])

    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, n_iter=1000, init='pca', learning_rate='auto')
    X_2d = tsne.fit_transform(X_combined)
    print("T-SNE 降维完成！")

    # 6. 绘图
    print("正在绘图...")
    plt.figure(figsize=(12, 10))

    # 绘制 Warm 点 (蓝色) - 把尺寸放大一点 (s=40)
    plt.scatter(X_2d[y_combined == 0, 0], X_2d[y_combined == 0, 1],
                c='tab:blue', label='Warm Embeddings (WMF + Content)',
                alpha=0.6, s=40, edgecolors='none')

    # 绘制 Cold 点 (橙色) - 把尺寸缩小一点 (s=10)
    plt.scatter(X_2d[y_combined == 1, 0], X_2d[y_combined == 1, 1],
                c='tab:orange', label='Cold Embeddings (Content Only)',
                alpha=0.9, s=10, edgecolors='none')

    plt.title(
        f'T-SNE Visualization of Item Embeddings in DropoutNet\n(Hidden Dims: {HIDDEN_DIMS}, Samples: {NUM_SAMPLES})',
        fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks([])
    plt.yticks([])

    save_path = 'tsne_visualization.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 可视化结果已保存为: {save_path}")
    plt.show()


if __name__ == "__main__":
    visualize_tsne()