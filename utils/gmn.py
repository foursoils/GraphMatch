import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

# ────────────────────────────────────────────────────────────────────────────
# GMN 单层传播
# ────────────────────────────────────────────────────────────────────────────
class GMNPropLayer(nn.Module):
    def __init__(self, node_in_dim: int, edge_dim: int, node_out_dim: int):
        """
        :param node_in_dim:  输入节点特征维度
        :param edge_dim:     边特征维度
        :param node_out_dim: 输出节点特征维度
        """
        super().__init__()
        self.node_out_dim = node_out_dim

        # 消息函数: [node_j || edge_ij] → node_out_dim
        self.message_net = nn.Sequential(
            nn.Linear(node_in_dim + edge_dim, node_out_dim),
            nn.LayerNorm(node_out_dim),
            nn.ReLU(),
        )

        # 更新函数: [node_i(in) || agg_msg(out) || (node_i - cross_i)(in)] → node_out_dim
        self.update_net = nn.Sequential(
            nn.Linear(node_in_dim + node_out_dim + node_in_dim, node_out_dim),
            nn.LayerNorm(node_out_dim),
            nn.ReLU(),
        )

    def _intra_propagate(self, x: torch.Tensor,
                         edge_index: torch.Tensor,
                         edge_attr: torch.Tensor) -> torch.Tensor:
        """图内消息传递"""
        num_nodes = x.size(0)
        if edge_index.size(1) == 0:
            return torch.zeros(num_nodes, self.node_out_dim, device=x.device)
        src, dst = edge_index[0], edge_index[1]
        msg_input = torch.cat([x[src], edge_attr], dim=-1)
        msgs = self.message_net(msg_input)
        hidden_dim = msgs.size(-1)
        agg = torch.zeros(num_nodes, hidden_dim, device=x.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(msgs), msgs)
        return agg

    def _cross_attention(self, x_query: torch.Tensor,
                         x_key: torch.Tensor,
                         batch_query: torch.Tensor,
                         batch_key: torch.Tensor) -> torch.Tensor:
        """跨图软注意力：同 batch 内配对图相互感知，跨样本隔离"""
        scale = x_query.size(-1) ** 0.5
        same_sample_mask = (batch_query.unsqueeze(1) == batch_key.unsqueeze(0))
        attn = torch.mm(x_query, x_key.t()) / scale
        attn = attn.masked_fill(~same_sample_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        return torch.mm(attn, x_key)

    def forward(self,
                x1: torch.Tensor, edge_index1: torch.Tensor, edge_attr1: torch.Tensor, batch1: torch.Tensor,
                x2: torch.Tensor, edge_index2: torch.Tensor, edge_attr2: torch.Tensor, batch2: torch.Tensor):
        agg1 = self._intra_propagate(x1, edge_index1, edge_attr1)
        agg2 = self._intra_propagate(x2, edge_index2, edge_attr2)
        cross1 = self._cross_attention(x1, x2, batch1, batch2)
        cross2 = self._cross_attention(x2, x1, batch2, batch1)
        x1_new = self.update_net(torch.cat([x1, agg1, x1 - cross1], dim=-1))
        x2_new = self.update_net(torch.cat([x2, agg2, x2 - cross2], dim=-1))
        return x1_new, x2_new


# ────────────────────────────────────────────────────────────────────────────
# GMN 编码器：复用 GMNPropLayer，输出节点级 + 图级向量
# ────────────────────────────────────────────────────────────────────────────
class GMNEncoder(nn.Module):
    def __init__(self,
                 node_input_dim: int = 1024,
                 edge_input_dim: int = 1024,
                 node_hidden_dim: int = 256,
                 num_prop_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()

        def build_proj(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
            )

        self.node_proj = build_proj(node_input_dim, node_hidden_dim)
        self.edge_proj = build_proj(edge_input_dim, node_hidden_dim)

        self.prop_layers = nn.ModuleList([
            GMNPropLayer(node_hidden_dim, node_hidden_dim, node_hidden_dim)
            for _ in range(num_prop_layers)
        ])

    def forward(self, batch):
        """
        返回:
          node_emb_c: claim 节点嵌入 [N_c, node_hidden_dim]
          node_emb_d: doc   节点嵌入 [N_d, node_hidden_dim]
          graph_global: 图级全局向量（claim+doc 池化平均）[B, node_hidden_dim]
          batch_c / batch_d: 节点归属 batch index（供 Cross-Attention 使用）
        """
        x1 = self.node_proj(batch.x_s)
        x2 = self.node_proj(batch.x_t)
        e1 = self.edge_proj(batch.edge_attr_s)
        e2 = self.edge_proj(batch.edge_attr_t)
        b1, b2 = batch.x_s_batch, batch.x_t_batch

        for layer in self.prop_layers:
            dx1, dx2 = layer(x1, batch.edge_index_s, e1, b1,
                             x2, batch.edge_index_t, e2, b2)
            x1 = x1 + dx1
            x2 = x2 + dx2

        g1 = global_mean_pool(x1, b1)   # [B, node_hidden_dim]
        g2 = global_mean_pool(x2, b2)   # [B, node_hidden_dim]
        graph_global = (g1 + g2) / 2.0  # [B, node_hidden_dim]

        return x1, x2, graph_global, b1, b2
