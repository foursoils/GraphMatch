"""
GraphCheck 模型
===============
架构：
  - 冻结的因果语言模型 (LLM)，作为主干解码器
  - 可训练的 GNN 图编码器，分别编码 claim 图和 doc 图
  - 可训练的投影层，将 GNN 输出对齐到 LLM 词嵌入空间
  - 推理时将图嵌入作为 Soft Prompt 前缀拼接到文本序列

参考来源（部分代码结构借鉴自）：
  G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering
  He et al. (2024), arXiv:2402.07630, https://github.com/XiaoxinHe/G-Retriever

  GraphCheck: Breaking Long-Term Text Barriers with Knowledge Graphs and LLMs
  （本实现为针对幻觉检测任务的适配版本）
"""

import os
import contextlib
import torch
import torch.nn as nn
from torch_geometric.utils import scatter
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.gnn import load_gnn_model


# ---------------------------------------------------------------------------
# 特殊 Token（Llama-2 / Llama-3 风格 Instruction 格式）
# ---------------------------------------------------------------------------

BOS      = '<s>[INST]'
EOS_USER = '[/INST]'
EOS      = '</s>'

IGNORE_INDEX = -100   # CrossEntropyLoss 忽略的标签值


# ---------------------------------------------------------------------------
# GraphCheck
# ---------------------------------------------------------------------------

class GraphCheck(nn.Module):
    """
    Graph-augmented LLM 幻觉检测模型。

    输入序列结构（Embedding 拼接顺序）：
        [BOS] [claim_graph_emb] [doc_graph_emb] [text_tokens] [EOS_USER] ([label_tokens] [EOS])
                                                                           ^^^^ 仅训练时包含 ^^^^
    """

    def __init__(self, config: dict):
        """
        Args:
            config: 对应 YAML 中 ablation.check 节点的配置字典。
        """
        super().__init__()

        model_cfg = config['model']
        gnn_cfg   = config['gnn']

        self.max_txt_len    = model_cfg.get('max_txt_len',    512)
        self.max_new_tokens = model_cfg.get('max_new_tokens', 16)

        # ---- 加载 LLM ----
        llm_path = model_cfg['llm_model_path']
        if not os.path.isabs(llm_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            cleaned = llm_path.lstrip('.').lstrip('/').lstrip('\\')
            llm_path = os.path.normpath(os.path.join(base_dir, cleaned))
        print(f"[Init] 加载 LLM: {llm_path} ...")

        # 判断是否在 DDP 分布式模式下
        local_rank = int(os.environ.get('LOCAL_RANK', -1))
        
        if local_rank != -1:
            # DDP 模式：将 LLM 完整加载到当前进程所负责的 GPU 卡上
            device_map = {"": local_rank}
            max_memory = None
        else:
            # 自动按卡分配显存（每张卡留 2 GiB 余量）
            num_devices = torch.cuda.device_count()
            max_memory  = {
                i: f"{max(torch.cuda.get_device_properties(i).total_memory // (1024**3) - 2, 2)}GiB"
                for i in range(num_devices)
            }
            device_map = 'auto'

        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False)
        self.tokenizer.pad_token_id  = 0
        self.tokenizer.padding_side  = 'left'

        llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=device_map,
            max_memory=max_memory,
        )
        # 冻结 LLM 所有参数，只有 GNN + Projector 参与训练
        for param in llm.parameters():
            param.requires_grad = False
        llm.gradient_checkpointing_enable()
        self.llm = llm
        print("[Init] LLM 加载完毕，参数已冻结。")

        # LLM 词嵌入层（只读引用，用于编码文本 token 和特殊 token）
        self.word_embedding = self.llm.model.get_input_embeddings()
        llm_emb_dim = self.word_embedding.weight.shape[1]

        # 记录设备 ID。DDP 下为 local_rank，否则为 0
        self.device_id = local_rank if local_rank != -1 else 0
        target_device = torch.device(f"cuda:{self.device_id}")

        # ---- 图编码器 ----
        gnn_cls = load_gnn_model(gnn_cfg.get('model_name', 'gt'))
        self.graph_encoder = gnn_cls(
            in_channels=gnn_cfg['in_dim'],
            hidden_channels=gnn_cfg['hidden_dim'],
            out_channels=gnn_cfg['hidden_dim'],
            num_layers=gnn_cfg['num_layers'],
            dropout=gnn_cfg.get('dropout', 0.1),
            num_heads=gnn_cfg.get('num_heads', 4),
        ).to(target_device)

        # ---- 对齐投影层 ----
        self.projector = nn.Sequential(
            nn.Linear(gnn_cfg['hidden_dim'], 2048),
            nn.Sigmoid(),
            nn.Linear(2048, llm_emb_dim),
        ).to(target_device)

        self.embed_dim = llm_emb_dim

        # ---- 动态决定 BOS / EOS_USER / EOS Token (支持 Qwen 的思考控制) ----
        # 对于当前算法（幻觉检测），模型思考能力不适用，在此强制关闭以保障小 token window 输出
        self.enable_thinking = False
        # 兼容本地路径名或 huggingface name 中的 qwen 关键字
        if 'qwen' in llm_path.lower() or 'qwen' in self.tokenizer.name_or_path.lower():
            self.bos_token = '<|im_start|>user\n'
            self.eos_token = '<|im_end|>'
            if self.enable_thinking:
                self.eos_user_token = '<|im_end|>\n<|im_start|>assistant\n<think>\n'
            else:
                self.eos_user_token = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        else:
            # 默认使用 Llama/Gemma 风格
            self.bos_token = '<s>[INST]'
            self.eos_user_token = '[/INST]'
            self.eos_token = '</s>'

    # -----------------------------------------------------------------------
    # 属性 & 工具方法
    # -----------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return torch.device(f"cuda:{self.device_id}")

    def maybe_autocast(self, dtype=torch.bfloat16):
        """在 GPU 上开启混合精度，CPU 上不做任何事。"""
        if self.device != torch.device('cpu'):
            return torch.amp.autocast(device_type='cuda', dtype=dtype)
        return contextlib.nullcontext()

    def print_trainable_params(self):
        """打印可训练参数数量及占比。"""
        trainable, total = 0, 0
        for _, p in self.named_parameters():
            n = p.numel()
            total += n
            if p.requires_grad:
                trainable += n
        print(f"可训练参数: {trainable:,} / 全部参数: {total:,} ({100 * trainable / total:.2f}%)")
        return trainable, total

    # -----------------------------------------------------------------------
    # 图编码
    # -----------------------------------------------------------------------

    def encode_graphs(self, data: dict):
        """
        对 claim 图和 doc 图分别进行 GNN 编码，并做全局平均池化。
        通过合并输入避免在 DDP 模式下对同一个 DDP 包装的 module 调用多次而导致 backward 冲突。

        Returns:
            claim_embeds: [B, gnn_hidden_dim]
            doc_embeds:   [B, gnn_hidden_dim]
        """
        claim_kg = data['claim_kg'].to(self.llm.device)
        doc_kg   = data['doc_kg'].to(self.llm.device)

        # 合并图节点和边特征
        num_nodes_claim = claim_kg.x.size(0)
        combined_x = torch.cat([claim_kg.x, doc_kg.x], dim=0)
        combined_edge_index = torch.cat([claim_kg.edge_index, doc_kg.edge_index + num_nodes_claim], dim=1)
        combined_edge_attr = torch.cat([claim_kg.edge_attr, doc_kg.edge_attr], dim=0)

        # 仅调用一次 graph_encoder
        combined_n_embeds, _ = self.graph_encoder(combined_x, combined_edge_index.long(), combined_edge_attr)

        claim_n_embeds = combined_n_embeds[:num_nodes_claim]
        doc_n_embeds = combined_n_embeds[num_nodes_claim:]

        # 全局平均池化：[N_nodes, D] -> [B, D]
        claim_embeds = (
            scatter(claim_n_embeds, claim_kg.batch, dim=0, reduce='mean')
            if claim_kg.batch is not None
            else claim_n_embeds.mean(dim=0, keepdim=True)
        )
        doc_embeds = (
            scatter(doc_n_embeds, doc_kg.batch, dim=0, reduce='mean')
            if doc_kg.batch is not None
            else doc_n_embeds.mean(dim=0, keepdim=True)
        )
        return claim_embeds, doc_embeds

    # -----------------------------------------------------------------------
    # 构建 Batch Embedding（训练 / 推理共用内部逻辑）
    # -----------------------------------------------------------------------

    def _build_batch_embeds(self, data: dict, include_label: bool = False):
        """
        将图嵌入 + 文本 token 拼接成 LLM 输入 Embedding 序列，并做 padding。

        Args:
            data:          来自 DataLoader 的一个 batch
            include_label: 训练时传 True，会在序列末尾拼接 label token

        Returns:
            inputs_embeds:   [B, max_L, D]
            attention_mask:  [B, max_L]
            label_input_ids: [B, max_L]  (仅 include_label=True 时有效)
        """
        texts  = self.tokenizer(data['text'],  add_special_tokens=False)
        labels = self.tokenizer(data['label'], add_special_tokens=False) if include_label else None

        # 特殊 token embedding
        _enc = lambda s: self.tokenizer(s, add_special_tokens=False)
        eos_ids      = _enc(self.eos_token).input_ids
        eos_user_ids = _enc(self.eos_user_token).input_ids
        bos_embeds   = self.word_embedding(
            self.tokenizer(self.bos_token, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.llm.device)
        )
        pad_embeds = self.word_embedding(
            torch.tensor(self.tokenizer.pad_token_id, device=self.llm.device)
        ).unsqueeze(0)   # [1, D]

        # 图编码
        claim_embeds, doc_embeds = self.encode_graphs(data)
        batch_size   = len(data['id'])

        # 合并并通过 projector，以避免在 DDP 模式下对同一个 DDP 包装的 module 调用多次而导致 backward 冲突
        combined_embeds = torch.cat([claim_embeds, doc_embeds], dim=0)
        combined_projected = self.projector(combined_embeds)
        claim_projected = combined_projected[:batch_size]
        doc_projected = combined_projected[batch_size:]

        batch_inputs_embeds  = []
        batch_attention_mask = []
        batch_label_ids      = []

        for i in range(batch_size):
            # 文本 token ids（截断）
            text_ids = texts.input_ids[i][:self.max_txt_len] + eos_user_ids

            if include_label:
                label_ids = labels.input_ids[i][:self.max_new_tokens] + eos_ids
                seq_ids   = text_ids + label_ids
            else:
                seq_ids = text_ids

            # 文本部分转 embedding
            seq_embeds = self.word_embedding(torch.tensor(seq_ids, device=self.llm.device))

            # 安全获取图嵌入（批大小对不上时用零向量兜底）
            c_emb = claim_projected[i].unsqueeze(0) if claim_projected.size(0) == batch_size \
                    else torch.zeros(1, self.embed_dim, device=self.llm.device)
            d_emb = doc_projected[i].unsqueeze(0) if doc_projected.size(0) == batch_size \
                    else torch.zeros(1, self.embed_dim, device=self.llm.device)

            # 最终输入序列: [BOS] [claim_graph] [doc_graph] [text...]
            full_embeds = torch.cat([bos_embeds, c_emb, d_emb, seq_embeds], dim=0)

            batch_inputs_embeds.append(full_embeds)
            batch_attention_mask.append([1] * full_embeds.shape[0])

            if include_label:
                prefix_len = full_embeds.shape[0] - len(label_ids)
                lbl = [IGNORE_INDEX] * prefix_len + label_ids
                batch_label_ids.append(lbl)

        # ---- 左 padding 对齐 ----
        max_len = max(x.shape[0] for x in batch_inputs_embeds)
        for i in range(batch_size):
            pad_len = max_len - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i]  = torch.cat([pad_embeds.repeat(pad_len, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_len + batch_attention_mask[i]
            if include_label:
                batch_label_ids[i] = [IGNORE_INDEX] * pad_len + batch_label_ids[i]

        inputs_embeds  = torch.stack(batch_inputs_embeds, dim=0).to(self.llm.device)
        attention_mask = torch.tensor(batch_attention_mask, device=self.llm.device)
        label_input_ids = (
            torch.tensor(batch_label_ids, device=self.llm.device)
            if include_label else None
        )
        return inputs_embeds, attention_mask, label_input_ids

    # -----------------------------------------------------------------------
    # 训练前向传播
    # -----------------------------------------------------------------------

    def forward(self, data: dict) -> torch.Tensor:
        """
        训练模式：计算 CrossEntropy Loss。

        Returns:
            loss: 标量张量
        """
        inputs_embeds, attention_mask, label_input_ids = self._build_batch_embeds(data, include_label=True)

        with self.maybe_autocast():
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=label_input_ids,
                return_dict=True,
            )
        return outputs.loss

    # -----------------------------------------------------------------------
    # 推理
    # -----------------------------------------------------------------------

    def inference(self, data: dict) -> dict:
        """
        推理模式：生成文本输出，返回预测结果字典。

        Returns:
            dict with keys: id, pred, label, text
        """
        inputs_embeds, attention_mask, _ = self._build_batch_embeds(data, include_label=False)

        with self.maybe_autocast():
            output_ids = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )

        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return {
            'id':    data['id'],
            'pred':  preds,
            'label': data['label'],
            'text':  data['text'],
        }
