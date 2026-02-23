import torch
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from model import DropoutNet
from dataset import load_data

# --- 配置参数 ---
BASE_DIR = './data/recsys2017.pub'
MODEL_PATH = 'dropoutnet_recsys_model.pth'
LATENT_DIM = 200
BATCH_SIZE = 1024
EVAL_BATCH_SIZE = 256
TOP_K = 100
HIDDEN_DIMS = [800, 800, 400]  # 与 train.py 保持一致

TEST_SCENARIOS = {
    'Warm Start': {
        'interactions': os.path.join(BASE_DIR, 'eval/warm/test_warm.csv'),
        'candidates': os.path.join(BASE_DIR, 'eval/warm/test_warm_item_ids.csv')
    },
    'User Cold Start': {
        'interactions': os.path.join(BASE_DIR, 'eval/warm/test_cold_user.csv'),
        'candidates': os.path.join(BASE_DIR, 'eval/warm/test_cold_user_item_ids.csv')
    },
    'Item Cold Start': {
        'interactions': os.path.join(BASE_DIR, 'eval/warm/test_cold_item.csv'),
        'candidates': os.path.join(BASE_DIR, 'eval/warm/test_cold_item_item_ids.csv')
    }
}


def generate_item_embeddings(model, v_pref, v_content, device, mask_wmf=False):
    model.eval()
    num_items = v_pref.size(0)
    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, num_items, BATCH_SIZE), desc=f"Generating Item Embs (Mask={mask_wmf})"):
            end_idx = min(i + BATCH_SIZE, num_items)

            batch_pref = v_pref[i:end_idx].to(device)
            if mask_wmf:
                batch_pref = torch.zeros_like(batch_pref)

            batch_content_sparse = v_content[i:end_idx]
            batch_content = torch.from_numpy(
                batch_content_sparse.toarray().astype(np.float32)
            ).to(device)

            input_cat = torch.cat([batch_pref, batch_content], dim=1)
            emb = model.item_network(input_cat)

            all_embeddings.append(emb)

    return torch.cat(all_embeddings, dim=0)


def evaluate_recall(model, test_csv, candidate_csv, u_pref, u_content, item_embs_matrix, device, mode='warm'):
    # 1. 读取测试数据
    df = pd.read_csv(test_csv, header=None, usecols=[0, 1], names=['uid', 'iid'])
    max_uid = u_pref.size(0)
    df = df[df['uid'] < max_uid]
    ground_truth = df.groupby('uid')['iid'].apply(set).to_dict()
    test_users = list(ground_truth.keys())

    # 2. 读取候选集
    candidate_iids = pd.read_csv(candidate_csv, header=None)[0].values
    print(f"   -> 测试用户数: {len(test_users)}, 候选物品数: {len(candidate_iids)}")

    # 3. 【核心修正】读取训练集以过滤历史交互
    train_csv_path = os.path.join(BASE_DIR, 'eval/warm/train.csv')
    train_df = pd.read_csv(train_csv_path, header=None, usecols=[0, 1], names=['uid', 'iid'])
    user_history = train_df.groupby('uid')['iid'].apply(set).to_dict()

    global_to_candidate = {item_id: idx for idx, item_id in enumerate(candidate_iids)}

    candidate_iids_tensor = torch.LongTensor(candidate_iids).to(device)
    candidate_item_embs = item_embs_matrix[candidate_iids_tensor]

    model.eval()
    hits = 0
    total_relevant = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(test_users), EVAL_BATCH_SIZE), desc=f"Eval {mode}", leave=False):
            batch_uids = test_users[i: min(i + EVAL_BATCH_SIZE, len(test_users))]

            batch_pref = u_pref[batch_uids].to(device)
            if mode == 'user_cold':
                batch_pref = torch.zeros_like(batch_pref)

            batch_content_sparse = u_content[batch_uids]
            batch_content = torch.from_numpy(
                batch_content_sparse.toarray().astype(np.float32)
            ).to(device)

            input_cat = torch.cat([batch_pref, batch_content], dim=1)
            user_embs = model.user_network(input_cat)

            scores = torch.matmul(user_embs, candidate_item_embs.t())

            # 【历史交互过滤】将用户在训练集中见过的物品分数设为极小值
            for idx, uid in enumerate(batch_uids):
                history = user_history.get(uid, set())
                mask_indices = [global_to_candidate[iid] for iid in history if iid in global_to_candidate]
                if mask_indices:
                    scores[idx, mask_indices] = -1e9

            _, topk_relative_indices = torch.topk(scores, k=TOP_K, dim=1)
            topk_relative_indices = topk_relative_indices.cpu().numpy()

            for idx, uid in enumerate(batch_uids):
                true_items = ground_truth[uid]
                relative_preds = topk_relative_indices[idx]
                pred_items = candidate_iids[relative_preds]

                hit_count = len(true_items.intersection(set(pred_items)))
                hits += hit_count
                total_relevant += len(true_items)

    recall = hits / total_relevant if total_relevant > 0 else 0
    return recall


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    _, u_pref, v_pref, u_content, v_content = load_data(BASE_DIR, LATENT_DIM)

    u_dim = u_content.shape[1]
    v_dim = v_content.shape[1]

    model = DropoutNet(
        u_latent_dim=LATENT_DIM,
        u_content_dim=u_dim,
        v_latent_dim=LATENT_DIM,
        v_content_dim=v_dim,
        hidden_dims=HIDDEN_DIMS
    ).to(device)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print("✅ 模型加载成功")
    else:
        print("❌ 模型文件不存在，请先运行 train.py")
        return

    item_embs_warm = generate_item_embeddings(model, v_pref, v_content, device, mask_wmf=False)
    item_embs_cold = generate_item_embeddings(model, v_pref, v_content, device, mask_wmf=True)

    print("\n" + "=" * 55)
    print(f"   Recall@{TOP_K} 评估报告 (基于候选集并过滤历史)")
    print("=" * 55)

    for scenario_name, paths in TEST_SCENARIOS.items():
        interactions_csv = paths['interactions']
        candidates_csv = paths['candidates']

        if not os.path.exists(interactions_csv) or not os.path.exists(candidates_csv):
            print(f"⚠️ 找不到文件，跳过: {scenario_name}")
            continue

        print(f"\n>>> 开始评估: {scenario_name}")

        mode = 'warm'
        target_item_matrix = item_embs_warm

        if 'User Cold' in scenario_name:
            mode = 'user_cold'
            target_item_matrix = item_embs_warm
        elif 'Item Cold' in scenario_name:
            mode = 'item_cold'
            target_item_matrix = item_embs_cold

        recall = evaluate_recall(
            model, interactions_csv, candidates_csv,
            u_pref, u_content,
            target_item_matrix,
            device,
            mode=mode
        )
        print(f"   【最终结果】 Recall@{TOP_K}: {recall:.4f}")


if __name__ == "__main__":
    main()