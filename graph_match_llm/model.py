"""
graph_match_llm - 模型模块
===========================
架构（路线 B：中间层 Cross-Attention 注入）：

  ┌─ Qwen3.5-4B (LoRA 微调) ─────────────────────────────────────┐
  │  Embedding                                                   │
  │  Layer 0 → 1 → ... → Layer k-1                              │
  │  Layer k  [full_attention] ──► + Cross-Attn(Q=text, KV=图节点)│  ← 注入点
  │  Layer k+1 → ... → Layer 31                                  │
  │  LM Head → 生成 CoT + "Therefore the answer is: Yes/No"       │
  └───────────────────────────────────────────────────────────────┘
                                          ▲
                         ┌──── GNN ────────┘
                         │  claim 图节点 [N_c, D]
                         │  doc   图节点 [N_d, D]
                         │  concat → [N_c+N_d, D]
                         │  → Projector → [N_c+N_d, llm_hidden]
                         └────────────────────────────────────────

可训练参数：
  - LLM LoRA adapter（q/k/v/o_proj）
  - GNN 编码器（全量）
  - GraphCrossAttnLayer（全量，插入到第 inject_layer 层之后）
  - Projector（全量）

冻结参数：
  - LLM 其余权重
"""

import os
import sys
import contextlib
import torch
import torch.nn as nn
from torch_geometric.utils import scatter
from torch_geometric.nn import global_mean_pool
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import get_peft_model, LoraConfig, TaskType
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False
    print("[Warning] peft 未安装，将不使用 LoRA（全量微调 LLM，显存压力大）。")

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

from utils.gmn import GMNEncoder

IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# 图节点投影层（GNN 输出 → LLM 隐空间）
# ---------------------------------------------------------------------------

class GraphProjector(nn.Module):
    def __init__(self, gnn_dim: int, llm_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(gnn_dim, gnn_dim * 2),
            nn.GELU(),
            nn.Linear(gnn_dim * 2, llm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# 中间层 Cross-Attention 模块
# ---------------------------------------------------------------------------

class GraphCrossAttnLayer(nn.Module):
    """
    在 LLM 某一层之后插入的 Cross-Attention 模块。

    Query  = 文本隐状态 h  [B, L, D_llm]
    Key/V  = 图节点嵌入   [B, N_graph, D_llm]（已由 Projector 投影）

    输出 = LayerNorm(h + tanh(alpha) * cross_attn(h, graph_nodes))
    """

    def __init__(self, llm_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=llm_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm    = nn.LayerNorm(llm_dim)
        self.dropout = nn.Dropout(dropout)
        # Learnable gating parameter initialized to 0
        self.alpha_micro = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        hidden_states: torch.Tensor,     # [B, L, D]
        graph_embeds:  torch.Tensor,     # [B, N, D]
        key_padding_mask: torch.Tensor = None,  # [B, N] True=padding
    ) -> torch.Tensor:
        attn_out, _ = self.cross_attn(
            query=hidden_states,
            key=graph_embeds,
            value=graph_embeds,
            key_padding_mask=key_padding_mask,
        )
        return self.norm(hidden_states + torch.tanh(self.alpha_micro) * self.dropout(attn_out))


# ---------------------------------------------------------------------------
# 主模型
# ---------------------------------------------------------------------------

class LLMGraphModel(nn.Module):
    """
    Qwen3.5-4B + LoRA + GNN + 中间层 Cross-Attention 注入。

    训练：forward(batch) → loss
    推理：inference(batch) → {'id', 'pred', 'label'}
    """

    def __init__(self, config: dict, device: torch.device = None):
        super().__init__()
        model_cfg  = config['model']
        gnn_cfg    = config.get('gmn', config.get('gnn', {}))  # 兼容两种 key
        lora_cfg   = config['lora']
        train_cfg  = config['training']

        self.max_txt_len      = model_cfg.get('max_txt_len',      1024)
        self.max_new_tokens   = model_cfg.get('max_new_tokens',   512)
        self.inject_layer     = model_cfg.get('inject_layer',     16)
        self.cross_attn_heads = model_cfg.get('cross_attn_heads', 8)
        self._aux_lambda      = train_cfg.get('aux_lambda',       0.3)

        # ---- 解析 LLM 路径 ----
        llm_path = model_cfg['llm_model_path']
        if not os.path.isabs(llm_path):
            cleaned  = llm_path.lstrip('.').lstrip('/').lstrip('\\')
            llm_path = os.path.normpath(os.path.join(_PROJ_ROOT, cleaned))

        # ---- Tokenizer ----
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'left'

        # ---- LLM 设备分发 ----
        if device is not None:
            _dev = device
            if _dev.type == 'cuda':
                device_map = {"": _dev.index if _dev.index is not None else 0}
            else:
                device_map = None
        else:
            local_rank = int(os.environ.get('LOCAL_RANK', -1))
            if local_rank != -1:
                device_map = {"": local_rank}
                _dev = torch.device(f"cuda:{local_rank}")
            else:
                num_gpus  = torch.cuda.device_count()
                max_mem   = {i: f"{torch.cuda.get_device_properties(i).total_memory // (1024**3) - 2}GiB"
                             for i in range(num_gpus)}
                device_map = 'auto'
                _dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print(f"[Init] 加载 LLM: {llm_path} (device_map={device_map})")
        llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )

        # LLM hidden size & layers/heads info
        base_cfg   = getattr(llm, 'config', None) or getattr(llm.base_model, 'config', None)
        text_cfg   = getattr(base_cfg, 'text_config', base_cfg)
        llm_dim    = text_cfg.hidden_size
        self.llm_dim = llm_dim
        self.num_heads = text_cfg.num_attention_heads
        num_layers = getattr(text_cfg, 'num_hidden_layers', 32)
        self.num_layers = num_layers

        # ---- LoRA ----
        if _PEFT_AVAILABLE:
            layers_to_transform = list(range(self.inject_layer, num_layers))
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_cfg.get('r', 16),
                lora_alpha=lora_cfg.get('lora_alpha', 32),
                lora_dropout=lora_cfg.get('lora_dropout', 0.05),
                target_modules=lora_cfg.get('target_modules', ['q_proj', 'k_proj', 'v_proj', 'o_proj']),
                bias="none",
                layers_to_transform=layers_to_transform,
            )
            llm = get_peft_model(llm, lora_config)
            print(f"[Init] LoRA 已应用 (仅作用于 Layer {self.inject_layer} 至 {num_layers-1})。")
            llm.print_trainable_parameters()
        else:
            # 没有 peft 时全量微调（仅调试用）
            print("[Init] 无 LoRA，LLM 全量可训练（仅调试）。")

        self.llm = llm

        # 记录主设备
        if _dev.type == 'cuda':
            self.device_id = _dev.index if _dev.index is not None else 0
        else:
            self.device_id = 0

        # ---- GMN 编码器（claim/doc 跨图对齐）----
        self.gmn = GMNEncoder(
            node_input_dim=gnn_cfg['in_dim'],
            edge_input_dim=gnn_cfg['in_dim'],
            node_hidden_dim=gnn_cfg['hidden_dim'],
            num_prop_layers=gnn_cfg.get('num_layers', 3),
            dropout=gnn_cfg.get('dropout', 0.3),
        ).to(_dev)

        # ---- Projector ----
        self.projector = GraphProjector(gnn_cfg['hidden_dim'], llm_dim).to(_dev)

        # ---- GMN 辅助分类头（直接监督 graph_global，给 GMN 强梯度信号）----
        gmn_dim = gnn_cfg['hidden_dim']
        self.gmn_cls_head = nn.Sequential(
            nn.Linear(gmn_dim, gmn_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(gmn_dim, 2),
        ).to(_dev)

        # ---- Cross-Attention 注入层 ----
        self.cross_attn_layer = GraphCrossAttnLayer(
            llm_dim=llm_dim,
            num_heads=self.cross_attn_heads,
            dropout=0.1,
        ).to(_dev)

        # ---- 宏观自注意力偏置注入 ----
        self.current_b_graph = None
        self.graph_to_head = nn.Linear(gmn_dim, self.num_heads, bias=False).to(_dev)
        self.gammas = nn.Parameter(torch.zeros(self.num_layers)).to(_dev)
        self._register_attention_bias_hooks()

        # ---- 注册 forward hook 到指定层 ----
        self._graph_kv: torch.Tensor = None   # 暂存图节点嵌入（hook 内访问）
        self._hook_handle = None
        self._register_inject_hook()

        # ---- 加载系统提示词 ----
        sys_path = os.path.join(_PROJ_ROOT, "prompts", "hallu_detect", "system_prompt.txt")
        if os.path.exists(sys_path):
            with open(sys_path, 'r', encoding='utf-8') as f:
                self.system_prompt = f.read().strip()
            print(f"[Init] 从 {sys_path} 加载系统提示词。")
        else:
            self.system_prompt = (
                "You are an expert fact-checker. "
                "Given a document and a claim, reason step by step and determine "
                "whether the document supports the claim."
            )

        print(f"[Init] 完成。LLM_dim={llm_dim}, 注入层={self.inject_layer}")

    # -----------------------------------------------------------------------
    # Hook 注入机制
    # -----------------------------------------------------------------------

    def _get_transformer_layers(self):
        """获取 LLM 底层 transformer 层列表（兼容 LoRA 包装）。"""
        base = self.llm
        # peft 包装后需要通过 base_model.model 访问
        for attr in ['base_model', 'model']:
            if hasattr(base, attr):
                base = getattr(base, attr)
        # Qwen3.5 结构: model.model.layers
        if hasattr(base, 'model') and hasattr(base.model, 'layers'):
            return base.model.layers
        if hasattr(base, 'layers'):
            return base.layers
        raise RuntimeError("无法找到 LLM transformer layers，请检查模型结构。")

    def _register_inject_hook(self):
        """在第 inject_layer 层注册 forward hook，完成 Cross-Attention 注入。"""
        layers = self._get_transformer_layers()
        k = min(self.inject_layer, len(layers) - 1)

        def _hook(module, input, output):
            # output 可能是 tuple（hidden, cache, ...）或纯 Tensor
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            if self._graph_kv is None:
                return output  # 图还没编码，跳过（推理前会设好）

            # graph_kv: [B, N, D]；需要与 hidden 同设备/dtype
            graph_kv = self._graph_kv.to(hidden.device, hidden.dtype)
            injected = self.cross_attn_layer(hidden, graph_kv)

            if isinstance(output, tuple):
                return (injected,) + output[1:]
            return injected

        self._hook_handle = layers[k].register_forward_hook(_hook)
        print(f"[Init] Cross-Attention hook 已注册到 Layer {k}。")

    def _register_attention_bias_hooks(self):
        """对 inject_layer 及之后的 self_attn 模块注册 attention bias patch。"""
        layers = self._get_transformer_layers()
        for idx in range(self.inject_layer, self.num_layers):
            layer = layers[idx]
            if hasattr(layer, 'self_attn'):
                self_attn_module = layer.self_attn
                self._patch_attention_forward(self_attn_module, idx)
        print(f"[Init] Attention Bias 注入已注册到 Layer {self.inject_layer} 至 {self.num_layers - 1}。")

    def _patch_attention_forward(self, self_attn_module, layer_idx):
        original_forward = self_attn_module.forward
        model_ref = self

        def new_forward(
            self,  # refers to self_attn_module
            hidden_states,
            position_embeddings = None,
            attention_mask = None,
            past_key_values = None,
            **kwargs
        ):
            if model_ref.current_b_graph is not None:
                B, L, _ = hidden_states.shape
                if attention_mask is not None:
                    S = attention_mask.shape[-1]
                elif past_key_values is not None:
                    if hasattr(past_key_values, "get_seq_len"):
                        S = past_key_values.get_seq_len(layer_idx) + L
                    else:
                        try:
                            S = past_key_values[layer_idx][0].shape[2] + L
                        except Exception:
                            S = L
                else:
                    S = L

                gamma = model_ref.gammas[layer_idx]
                b_bias = gamma * model_ref.current_b_graph.unsqueeze(-1).unsqueeze(-1)
                b_bias = b_bias.to(device=hidden_states.device, dtype=hidden_states.dtype)

                if attention_mask is not None:
                    b_bias = b_bias.expand(B, model_ref.num_heads, L, S)
                    attention_mask = (attention_mask + b_bias).contiguous()
                else:
                    if L > 1:
                        causal = torch.full((L, S), fill_value=float("-inf"), device=hidden_states.device, dtype=hidden_states.dtype)
                        causal = torch.triu(causal, diagonal=1)
                        attention_mask = (causal.unsqueeze(0).unsqueeze(1) + b_bias).contiguous()
                    else:
                        attention_mask = b_bias.expand(B, model_ref.num_heads, L, S).contiguous()

            return original_forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs
            )

        import types
        self_attn_module.forward = types.MethodType(new_forward, self_attn_module)

    # -----------------------------------------------------------------------
    # 图编码
    # -----------------------------------------------------------------------

    def _encode_graphs(self, data: dict):
        """
        GMN 编码 claim/doc PairData，claim 与 doc 节点跨图互相对齐。

        返回:
            padded      : [B, N_max, llm_dim]  供 Cross-Attention 注入
            graph_global: [B, gmn_dim]          供辅助分类头使用
        """
        _dev = torch.device(f"cuda:{self.device_id}")
        pair = data['graph_pair'].to(_dev)

        # GMNEncoder: claim/doc 跨图消息传递
        # node_c [N_c, gmn_dim], node_d [N_d, gmn_dim],
        # graph_global [B, gmn_dim], batch_c, batch_d
        node_c, node_d, graph_global, batch_c, batch_d = self.gmn(pair)

        batch_size = len(data['id'])
        # 计算两图全局特征向量之差（Macro-Level 注入）
        g_c = global_mean_pool(node_c, batch_c, size=batch_size)  # [B, gmn_dim]
        g_d = global_mean_pool(node_d, batch_d, size=batch_size)  # [B, gmn_dim]
        delta_h_g = g_c - g_d                                     # [B, gmn_dim]

        # 投影并在 model 上暂存，供 self_attn.forward 访问
        self.current_b_graph = self.graph_to_head(delta_h_g)       # [B, num_heads]

        # 合并节点 → Projector → LLM 维度
        node_all  = self.projector(torch.cat([node_c, node_d], dim=0))  # [N_c+N_d, llm_dim]
        batch_all = torch.cat([batch_c, batch_d], dim=0)

        per_sample = []
        for i in range(batch_size):
            nodes_i = node_all[batch_all == i]
            per_sample.append(nodes_i if nodes_i.size(0) > 0
                               else torch.zeros(1, self.llm_dim, device=_dev))

        max_n  = max(s.size(0) for s in per_sample)
        padded = torch.zeros(batch_size, max_n, self.llm_dim, device=_dev)
        for i, s in enumerate(per_sample):
            padded[i, :s.size(0)] = s

        return padded, graph_global  # [B,N_max,llm_dim], [B,gmn_dim]

    # -----------------------------------------------------------------------
    # 辅助：Qwen chat template
    # -----------------------------------------------------------------------

    def _make_chat_messages(self, instruction: str):
        return [
            {"role": "system",    "content": self.system_prompt},
            {"role": "user",      "content": instruction},
        ]

    def _apply_chat_template(self, instructions: list, add_generation_prompt: bool = True):
        """将 instruction 列表通过 tokenizer chat template 转为字符串列表。"""
        prompts = []
        for inst in instructions:
            msgs = self._make_chat_messages(inst)
            try:
                txt = self.tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            except Exception:
                # Fallback：无 chat template
                txt = f"[INST] {inst} [/INST]"
            prompts.append(txt)
        return prompts

    # -----------------------------------------------------------------------
    # autocast helper
    # -----------------------------------------------------------------------

    def maybe_autocast(self, dtype=torch.bfloat16):
        if torch.cuda.is_available():
            return torch.amp.autocast(device_type='cuda', dtype=dtype)
        return contextlib.nullcontext()

    # -----------------------------------------------------------------------
    # 训练前向
    # -----------------------------------------------------------------------

    def forward(self, batch: dict) -> torch.Tensor:
        """
        训练模式：拼接 prompt + target，计算 SFT loss。

        Loss 仅在 target（CoT + 答案）部分计算，instruction 部分 mask 掉。
        """
        _dev = torch.device(f"cuda:{self.device_id}")

        # 1. 图编码，暂存给 hook 用
        with self.maybe_autocast():
            self._graph_kv, graph_global = self._encode_graphs(batch)

        # 2. 构建 full prompt（instruction + target）
        prompts = self._apply_chat_template(batch['instruction'], add_generation_prompt=True)
        targets = batch['target']  # list of str

        full_texts = [p + t for p, t in zip(prompts, targets)]

        # 3. Tokenize
        enc_full = self.tokenizer(
            full_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_txt_len + self.max_new_tokens,  # instruction + CoT
        ).to(_dev)

        enc_prompt = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_txt_len,
        ).to(_dev)

        input_ids      = enc_full.input_ids
        attention_mask = enc_full.attention_mask

        # 4. 构建 labels：只对 target 部分算 loss
        # tokenizer.padding_side='left'，序列格式：[PAD...PAD | prompt | target]
        # 需要用 full 序列长度定位 prompt 的实际结束位置，不能直接用 :prompt_len
        labels = input_ids.clone()
        seq_len = input_ids.shape[1]
        full_lens   = enc_full.attention_mask.sum(dim=1).tolist()    # 每条样本的实际总长（不含 PAD）
        prompt_lens = enc_prompt.attention_mask.sum(dim=1).tolist()  # 每条样本的实际 prompt 长
        for i, (full_len, prompt_len) in enumerate(zip(full_lens, prompt_lens)):
            # 左填充下 prompt 结束位置 = seq_len - full_len + prompt_len
            prompt_end = seq_len - int(full_len) + int(prompt_len)
            labels[i, :prompt_end] = IGNORE_INDEX
        labels[attention_mask == 0] = IGNORE_INDEX

        # 5. LLM forward（hook 内会注入图信息）
        with self.maybe_autocast():
            outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )

        self._graph_kv = None  # 清理
        self.current_b_graph = None  # 清理

        # 6. 辅助分类 loss（GMN graph_global → 直接预测 label）
        lm_loss  = outputs.loss
        aux_loss = torch.tensor(0.0, device=lm_loss.device)
        if 'label' in batch and graph_global is not None:
            label_tensor = torch.tensor(batch['label'], dtype=torch.long,
                                        device=graph_global.device)
            logits   = self.gmn_cls_head(graph_global.to(lm_loss.device))
            aux_loss = nn.functional.cross_entropy(logits, label_tensor)

        aux_lambda = getattr(self, '_aux_lambda', 0.3)
        return lm_loss + aux_lambda * aux_loss, lm_loss.detach(), aux_loss.detach()

    # -----------------------------------------------------------------------
    # 推理
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def inference(self, batch: dict) -> dict:
        """
        推理模式：生成输出文本，返回预测结果。
        """
        _dev = torch.device(f"cuda:{self.device_id}")

        with self.maybe_autocast():
            self._graph_kv, _ = self._encode_graphs(batch)

        prompts = self._apply_chat_template(batch['instruction'], add_generation_prompt=True)
        enc = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_txt_len,
        ).to(_dev)

        with self.maybe_autocast():
            output_ids = self.llm.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        self._graph_kv = None
        self.current_b_graph = None

        # 只截取新生成的 token
        new_ids = output_ids[:, enc.input_ids.shape[1]:]
        preds   = self.tokenizer.batch_decode(new_ids, skip_special_tokens=True)

        return {
            'id':    batch['id'],
            'pred':  preds,
            'label': batch['label'],
            'text':  batch['instruction'],
        }

    # -----------------------------------------------------------------------
    # 工具
    # -----------------------------------------------------------------------

    def print_trainable_params(self):
        trainable, total = 0, 0
        for _, p in self.named_parameters():
            n = p.numel()
            total += n
            if p.requires_grad:
                trainable += n
        pct = 100 * trainable / total if total > 0 else 0
        print(f"可训练参数: {trainable:,} / 全部参数: {total:,} ({pct:.2f}%)")
        return trainable, total

    def remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
