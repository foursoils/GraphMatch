"""
图神经网络编码器
================
提供两种 GNN 变体：
  - GAT：图注意力网络
  - GraphTransformer：基于 Transformer 的图卷积网络
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GATConv


# ---------------------------------------------------------------------------
# Graph Transformer
# ---------------------------------------------------------------------------

class GraphTransformer(torch.nn.Module):
    """多层 TransformerConv，带 BatchNorm 和 Dropout。"""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=4):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns   = torch.nn.ModuleList()

        # 第一层
        self.convs.append(TransformerConv(
            in_channels=in_channels,
            out_channels=hidden_channels // num_heads,
            heads=num_heads,
            edge_dim=in_channels,
            dropout=dropout,
        ))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels // num_heads,
                heads=num_heads,
                edge_dim=in_channels,
                dropout=dropout,
            ))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        # 最后一层
        self.convs.append(TransformerConv(
            in_channels=hidden_channels,
            out_channels=out_channels // num_heads,
            heads=num_heads,
            edge_dim=in_channels,
            dropout=dropout,
        ))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index=edge_index, edge_attr=edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index=edge_index, edge_attr=edge_attr)
        return x, edge_attr


# ---------------------------------------------------------------------------
# GAT
# ---------------------------------------------------------------------------

class GAT(torch.nn.Module):
    """多层图注意力网络。"""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=4):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns   = torch.nn.ModuleList()

        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, concat=False))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.convs.append(GATConv(hidden_channels, out_channels, heads=num_heads, concat=False))
        self.dropout = dropout
        self.attn_weights = None

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        for i, conv in enumerate(self.convs[:-1]):
            x, (_, attn) = conv(x, edge_index=edge_index, edge_attr=edge_attr, return_attention_weights=True)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x, (_, attn) = self.convs[-1](x, edge_index=edge_index, edge_attr=edge_attr, return_attention_weights=True)
        self.attn_weights = attn
        return x, edge_attr


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------

GNN_MODELS = {
    'gat': GAT,
    'gt':  GraphTransformer,
}


def load_gnn_model(name: str):
    """按名称加载 GNN 类，不存在时抛出 ValueError。"""
    if name not in GNN_MODELS:
        raise ValueError(f"未知 GNN 模型: '{name}'，可选: {list(GNN_MODELS.keys())}")
    return GNN_MODELS[name]
