import torch
import torch.nn as nn


class DropoutNet(nn.Module):
    def __init__(self,
                 u_latent_dim, u_content_dim,
                 v_latent_dim, v_content_dim,
                 hidden_dims=[800, 800, 400],
                 dropout_rate=0.5):
        super(DropoutNet, self).__init__()
        self.dropout_rate = dropout_rate

        user_input_dim = u_latent_dim + u_content_dim
        self.user_network = self._build_mlp(user_input_dim, hidden_dims, u_latent_dim)

        item_input_dim = v_latent_dim + v_content_dim
        self.item_network = self._build_mlp(item_input_dim, hidden_dims, v_latent_dim)

    def _build_mlp(self, input_dim, hidden_dims, output_dim):
        """
        构建遵循论文的最优三层金字塔结构
        """
        layers = []
        curr_dim = input_dim

        for h_dim in hidden_dims:
            linear_layer = nn.Linear(curr_dim, h_dim)
            # 【关键修复】: 针对 Tanh 必须使用 Xavier/Glorot 初始化，防止梯度消失
            nn.init.xavier_uniform_(linear_layer.weight)

            layers.append(linear_layer)
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.Tanh())

            # 【关键修复】: 移除内部 Dropout，因为 Input Dropout 已经提供了足够的正则化 [cite: 175]
            # layers.append(nn.Dropout(0.2))

            curr_dim = h_dim

        final_layer = nn.Linear(curr_dim, output_dim)
        nn.init.xavier_uniform_(final_layer.weight)
        layers.append(final_layer)

        return nn.Sequential(*layers)

    def apply_input_dropout(self, preference_emb):
        if self.training:
            mask = torch.bernoulli(torch.full((preference_emb.size(0), 1), 1 - self.dropout_rate))
            mask = mask.to(preference_emb.device)
            return preference_emb * mask
        else:
            return preference_emb

    def forward(self, u_pref, u_content, v_pref, v_content, apply_dropout=True):
        if apply_dropout:
            u_pref_in = self.apply_input_dropout(u_pref)
            v_pref_in = self.apply_input_dropout(v_pref)
        else:
            u_pref_in = u_pref
            v_pref_in = v_pref

        u_concat = torch.cat([u_pref_in, u_content], dim=1)
        v_concat = torch.cat([v_pref_in, v_content], dim=1)

        u_hat = self.user_network(u_concat)
        v_hat = self.item_network(v_concat)

        scores = (u_hat * v_hat).sum(dim=1)
        return scores, u_hat, v_hat