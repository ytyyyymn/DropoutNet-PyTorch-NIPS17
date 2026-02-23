🛠️ 环境依赖 (Requirements)
Python 3.8+

PyTorch 1.12+ (支持 CUDA)

NumPy

SciPy

Pandas

scikit-learn

Matplotlib

tqdm

🚀 快速开始 (Quick Start)
1. 数据准备
请确保 recsys2017.pub 数据集已正确放置在 data/ 目录下，并且保持特征文件为 LibSVM 的稀疏格式 (_0based.txt)。

2. 模型训练
直接运行训练脚本。脚本内置了 [800, 800, 400] 的三层金字塔结构、Xavier 初始化机制以及防梯度爆炸的裁剪策略。

训练结束后，会自动保存 dropoutnet_recsys_model.pth 权重文件，并生成一张 Loss 曲线图 training_loss_curve.png。

3. 指标评估
运行测试脚本，模型将在严格划分的候选集上进行测试，并自动过滤训练集中的历史交互记录。

4. 隐空间可视化 (t-SNE)
一键生成冷热向量的空间映射对比图（打靶荷包蛋图），验证特征提取效果。

📊 评估结果 (Evaluation Results)
在 XING 数据集上的 Recall@100 表现：

结论: User Cold Start 指标不仅完美对齐，甚至实现了微弱的超越。Warm Start 高度贴近 SOTA 水平。极大地证明了代码中特征融合与推断变换逻辑的正确性。

💡 踩坑与心得 (Implementation Details)
网络深度与初始化: 处理上千维极度稀疏的 Content Feature 时，必须采用深层网络配合 Tanh 激活函数，且强依赖 Xavier Uniform 初始化，否则极易发生梯度消失或 NaN 爆炸。

评估候选集: 必须严格使用 test_xxx_item_ids.csv 进行测试。如果在 130 万全量物品池中盲测，或者忘记 Mask 掉 train.csv 的历史交互，会导致 Recall 指标呈现断崖式下跌。