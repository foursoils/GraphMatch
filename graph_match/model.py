"""
graph_match - 模型模块（LLM 版 GraphMatch）
============================================
默认 Backbone：Gemma-3-1b-it + LoRA

  Layer k:
    Self-Attn  ── Macro: 图差异 Δg → SA logits bias（Softmax 前）
         ↓ H^(k)
    Cross-Attn ── Micro: Q=H^(k), KV=图节点（SA 后、FFN 前）
         ↓
    FFN → Layer k+1 … → LM Head → 生成 0/1

可训练：LoRA（layer k…N-1）、GMN、Projector、Macro bias、Micro Cross-Attn
冻结：LLM 基座权重
"""

import os
import sys
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from transformers import AutoModelForCausalLM, AutoTokenizer

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

from utils.path_utils import log_rank0, configure_dist_process_logging, resolve_path

try:
    from peft import get_peft_model, LoraConfig, TaskType
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False
    log_rank0("[Warning] peft 未安装，将不使用 LoRA（全量微调 LLM，显存压力大）。")

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

class GraphMacroBias(nn.Module):
    """
    Macro 通路：图级差异 Δg 投影为 Self-Attention logits 的 non-uniform bias。

    对每个 head h：u^h = Δg W_G^h，b_{·j}^h = (u^h · k_j^h) / sqrt(d_h)
    返回 [B, H, 1, L]，在 Softmax 前加到 attention logits。
    """

    _GATE_EPS = 1e-6

    def __init__(self, gmn_dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.head_dim  = head_dim
        self.num_heads = num_heads
        self.head_projs = nn.ModuleList([
            nn.Linear(gmn_dim, head_dim, bias=False) for _ in range(num_heads)
        ])
        self.alpha_macro = nn.Parameter(torch.zeros(1))

    def compute_bias(self, delta_g: torch.Tensor, key_states: torch.Tensor):
        """key_states: [B, H, L, D]（RoPE 后）；返回 [B, H, 1, L] 或 None。"""
        gate = torch.tanh(self.alpha_macro)
        if gate.abs() < self._GATE_EPS or delta_g is None:
            return None
        _, H, _, D = key_states.shape
        scale = D ** -0.5
        per_head = []
        for h in range(H):
            u = self.head_projs[h](delta_g)           # [B, D]
            k = key_states[:, h, :, :]              # [B, L, D]
            per_head.append(torch.einsum('bd,bld->bl', u, k) * scale)
        bias = torch.stack(per_head, dim=1)           # [B, H, L]
        return gate * bias.unsqueeze(2)               # [B, H, 1, L]


class GraphCrossAttnLayer(nn.Module):
    """
    Micro 通路：Self-Attention 输出 H^(k) 上的 Cross-Attention。

    Query  = H^(k)  [B, L, D_llm]
    Key/V  = 图节点  [B, N, D_llm]（Projector 后）

    输出 = LN(H^(k) + tanh(alpha_micro) * CrossAttn(...))；alpha≈0 时恒等返回。
    """

    _GATE_EPS = 1e-6

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
        gate = torch.tanh(self.alpha_micro)
        if gate.abs() < self._GATE_EPS:
            return hidden_states

        attn_out, _ = self.cross_attn(
            query=hidden_states,
            key=graph_embeds,
            value=graph_embeds,
            key_padding_mask=key_padding_mask,
        )
        return self.norm(hidden_states + gate * self.dropout(attn_out))


# ---------------------------------------------------------------------------
# 主模型
# ---------------------------------------------------------------------------

class LLMGraphModel(nn.Module):
    """
    Gemma-3-1b-it + LoRA + GMN + 双尺度中间层注入（Macro SA bias + Micro Cross-Attn）。

    训练：forward(batch) → (total_loss, lm_loss, aux_loss)
    推理：inference(batch) → {'id', 'pred', 'label'}
    """

    def __init__(self, config: dict, device: torch.device = None, apply_lora: bool = True):
        super().__init__()
        model_cfg  = config['model']
        gmn_cfg    = config['gmn']
        lora_cfg   = config['lora']
        train_cfg  = config['training']

        self.max_txt_len      = model_cfg.get('max_txt_len', 1024)
        self.max_new_tokens   = model_cfg.get('max_new_tokens', 6)
        self.inject_layer     = model_cfg.get('inject_layer', 12)
        self.cross_attn_heads = model_cfg.get('cross_attn_heads', 8)

        self.aux_lambda_max           = train_cfg.get('aux_lambda', 0.5)
        self.aux_lambda_start         = train_cfg.get('aux_lambda_start', 0.2)
        self.aux_lambda_warmup_epochs = train_cfg.get('aux_lambda_warmup_epochs', 2)
        self.aux_cosine_target        = train_cfg.get('aux_cosine_target', 0.3)
        self._current_epoch           = 1

        llm_path = resolve_path(model_cfg['llm_model_path'])

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

        log_rank0(f"[Init] 加载 LLM: {llm_path} (device_map={device_map})")
        configure_dist_process_logging()
        llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            dtype=torch.bfloat16,
            attn_implementation=model_cfg.get('attn_implementation', 'eager'),
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
        self.head_dim = getattr(text_cfg, 'head_dim', llm_dim // self.num_heads)

        # ---- LoRA ----
        # 推理时 apply_lora=False，由 evaluate.load_checkpoint 从 adapter 目录加载
        if _PEFT_AVAILABLE and apply_lora:
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
            log_rank0(f"[Init] LoRA 已应用 (仅作用于 Layer {self.inject_layer} 至 {num_layers-1})。")
        elif not apply_lora:
            log_rank0("[Init] 推理模式：跳过 LoRA 初始化，等待 load_checkpoint 加载 adapter。")
        else:
            # 没有 peft 时全量微调（仅调试用）
            log_rank0("[Init] 无 LoRA，LLM 全量可训练（仅调试）。")

        self.llm = llm

        # 记录主设备
        if _dev.type == 'cuda':
            self.device_id = _dev.index if _dev.index is not None else 0
        else:
            self.device_id = 0

        # ---- GMN 编码器（claim/doc 跨图对齐）----
        self.gmn = GMNEncoder(
            node_input_dim=gmn_cfg['in_dim'],
            edge_input_dim=gmn_cfg['in_dim'],
            node_hidden_dim=gmn_cfg['hidden_dim'],
            num_prop_layers=gmn_cfg.get('num_layers', 3),
            dropout=gmn_cfg.get('dropout', 0.3),
        ).to(_dev)

        self.projector = GraphProjector(gmn_cfg['hidden_dim'], llm_dim).to(_dev)

        self.cross_attn_layer = GraphCrossAttnLayer(
            llm_dim=llm_dim,
            num_heads=self.cross_attn_heads,
            dropout=0.1,
        ).to(_dev)

        # ---- Macro：Self-Attention logits bias ----
        self.current_delta_h_g = None
        self.macro_bias = GraphMacroBias(
            gmn_dim=gmn_cfg['hidden_dim'],
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        ).to(_dev)

        # 图注入模块与 LLM 对齐 bfloat16
        if _dev.type == 'cuda':
            graph_dtype = torch.bfloat16
            self.projector = self.projector.to(dtype=graph_dtype)
            self.cross_attn_layer = self.cross_attn_layer.to(dtype=graph_dtype)
            self.macro_bias = self.macro_bias.to(dtype=graph_dtype)

        # ---- 注册 Macro / Micro 注入 hook ----
        self._graph_kv: torch.Tensor = None
        self._hook_handles = []
        self._orig_attn_forward = None
        self._register_inject_hooks()

        # ---- 加载系统提示词 ----
        prompt_rel = model_cfg.get(
            'system_prompt_path',
            'prompts/hallu_detect/system_prompt.txt',
        )
        sys_path = resolve_path(prompt_rel)
        if os.path.exists(sys_path):
            with open(sys_path, 'r', encoding='utf-8') as f:
                self.system_prompt = f.read().strip()
            log_rank0(f"[Init] 从 {sys_path} 加载系统提示词。")
        else:
            self.system_prompt = (
                "You are an expert fact-checker. "
                "Given a document and a claim, reason step by step and determine "
                "whether the document supports the claim."
            )

        log_rank0(f"[Init] 完成。LLM_dim={llm_dim}, 注入层={self.inject_layer}")

    # -----------------------------------------------------------------------
    # Plan-D / Plan-D-v2 辅助 loss
    # -----------------------------------------------------------------------

    def _get_aux_lambda(self) -> float:
        warmup = max(int(self.aux_lambda_warmup_epochs), 1)
        if self._current_epoch >= warmup:
            return float(self.aux_lambda_max)
        t = (self._current_epoch - 1) / warmup
        return float(self.aux_lambda_start + (self.aux_lambda_max - self.aux_lambda_start) * t)

    def _compute_aux_loss(self, g_c: torch.Tensor, g_d: torch.Tensor, labels: list) -> torch.Tensor:
        label_tensor = torch.tensor(labels, dtype=torch.long, device=g_c.device)
        sim = F.cosine_similarity(g_c.float(), g_d.float(), dim=-1)
        signed_target = (2.0 * label_tensor.float() - 1.0) * self.aux_cosine_target
        return F.mse_loss(sim, signed_target)

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
        # Qwen3 / Gemma3+PEFT: model.model.layers；Gemma3 裸模型: model.layers
        if hasattr(base, 'model') and hasattr(base.model, 'layers'):
            return base.model.layers
        if hasattr(base, 'layers'):
            return base.layers
        raise RuntimeError("无法找到 LLM transformer layers，请检查模型结构。")

    def _get_self_attn(self, layer: nn.Module) -> nn.Module:
        if hasattr(layer, 'self_attn'):
            return layer.self_attn
        raise RuntimeError("Decoder layer 缺少 self_attn，请检查 Gemma 模型结构。")

    @staticmethod
    def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        if n_rep == 1:
            return hidden_states
        b, n_kv, s, d = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(b, n_kv, n_rep, s, d)
        return hidden_states.reshape(b, n_kv * n_rep, s, d)

    def _self_attn_with_macro(self, attn_module, hidden_states, position_embeddings,
                               attention_mask, past_key_values, **kwargs):
        """Gemma3/Llama 风格 self-attn，在 Softmax 前注入 Macro bias。"""
        from transformers.models.gemma3.modeling_gemma3 import (
            apply_rotary_pos_emb,
            eager_attention_forward,
        )
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn_module.head_dim)

        query_states = attn_module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states   = attn_module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = attn_module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if hasattr(attn_module, 'q_norm'):
            query_states = attn_module.q_norm(query_states)
        if hasattr(attn_module, 'k_norm'):
            key_states = attn_module.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {k: kwargs[k] for k in ('cache_position',) if k in kwargs}
            key_states, value_states = past_key_values.update(
                key_states, value_states, attn_module.layer_idx, cache_kwargs
            )

        num_kv_heads = getattr(attn_module, 'num_key_value_heads', self.num_heads)
        num_kv_groups = self.num_heads // num_kv_heads
        key_for_bias = self._repeat_kv(key_states, num_kv_groups)
        macro_bias = self.macro_bias.compute_bias(self.current_delta_h_g, key_for_bias)
        if macro_bias is not None:
            if attention_mask is None:
                attention_mask = macro_bias
            else:
                attention_mask = attention_mask + macro_bias.to(attention_mask.dtype)

        attn_impl = getattr(attn_module.config, '_attn_implementation', 'eager')
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            attn_impl, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            attn_module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=attn_module.attention_dropout if attn_module.training else 0.0,
            scaling=getattr(attn_module, 'scaling', self.head_dim ** -0.5),
            sliding_window=getattr(attn_module, 'sliding_window', None),
            softcap=getattr(attn_module, 'attn_logit_softcapping', None),
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_module.o_proj(attn_output)
        return attn_output, attn_weights

    def _make_patched_self_attn_forward(self, attn_module, original_forward):
        def patched_forward(hidden_states, *args, **kwargs):
            use_macro = (
                self.current_delta_h_g is not None
                and torch.tanh(self.macro_bias.alpha_macro).abs() >= GraphMacroBias._GATE_EPS
            )
            if not use_macro:
                return original_forward(hidden_states, *args, **kwargs)
            try:
                position_embeddings = kwargs.get('position_embeddings')
                if position_embeddings is None and len(args) >= 1:
                    position_embeddings = args[0]
                attention_mask = kwargs.get('attention_mask')
                past_key_values = kwargs.get('past_key_values')
                return self._self_attn_with_macro(
                    attn_module,
                    hidden_states,
                    position_embeddings,
                    attention_mask,
                    past_key_values,
                    **kwargs,
                )
            except Exception:
                return original_forward(hidden_states, *args, **kwargs)
        return patched_forward

    def _register_inject_hooks(self):
        """在第 inject_layer 层注册 Macro（SA 内）与 Micro（SA 后）注入。"""
        layers = self._get_transformer_layers()
        k = min(self.inject_layer, len(layers) - 1)
        attn = self._get_self_attn(layers[k])

        # Macro：patch self_attn.forward，在 Softmax 前加 logits bias
        self._orig_attn_forward = attn.forward
        attn.forward = self._make_patched_self_attn_forward(attn, self._orig_attn_forward)

        # Micro：self_attn 输出 H^(k) 上做 Cross-Attention（SA 后、FFN 前）
        def _micro_hook(module, input, output):
            if self._graph_kv is None:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            graph_kv = self._graph_kv.to(hidden.device, hidden.dtype)
            injected = self.cross_attn_layer(hidden, graph_kv)
            if isinstance(output, tuple):
                return (injected,) + output[1:]
            return injected

        self._hook_handles.append(attn.register_forward_hook(_micro_hook))
        log_rank0(
            f"[Init] Macro(SA bias) + Micro(Cross-Attn) 已注册到 Layer {k} self_attn。"
        )

    # -----------------------------------------------------------------------
    # 图编码
    # -----------------------------------------------------------------------

    def _encode_graphs(self, data: dict):
        """
        GMN 编码 claim/doc PairData，claim 与 doc 节点跨图互相对齐。

        返回:
            padded: [B, N_max, llm_dim]  供 Cross-Attention 注入
            g_c   : [B, gmn_dim]          claim 图全局均值，供辅助分类头使用
            g_d   : [B, gmn_dim]          doc   图全局均值，供辅助分类头使用
        """
        _dev = torch.device(f"cuda:{self.device_id}")
        pair = data['graph_pair'].to(_dev)

        # GMNEncoder: claim/doc 跨图消息传递
        # node_c [N_c, gmn_dim], node_d [N_d, gmn_dim],
        # graph_global [B, gmn_dim], batch_c, batch_d
        node_c, node_d, _, batch_c, batch_d = self.gmn(pair)

        batch_size = len(data['id'])
        # 计算两图全局特征向量之差（Macro-Level 注入）
        g_c = global_mean_pool(node_c, batch_c, size=batch_size)  # [B, gmn_dim]
        g_d = global_mean_pool(node_d, batch_d, size=batch_size)  # [B, gmn_dim]
        delta_h_g = g_c - g_d                                     # [B, gmn_dim]

        # 在 model 上暂存，供 hook 访问
        self.current_delta_h_g = delta_h_g                         # [B, gmn_dim]

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

        return padded, g_c, g_d  # [B,N_max,llm_dim], [B,gmn_dim], [B,gmn_dim]

    # -----------------------------------------------------------------------
    # 辅助：Qwen chat template
    # -----------------------------------------------------------------------

    def _make_chat_messages(self, instruction: str):
        return [
            {"role": "system",    "content": self.system_prompt},
            {"role": "user",      "content": instruction},
        ]

    def _apply_chat_template(self, instructions: list, add_generation_prompt: bool = True):
        """将 instruction 列表转为 Gemma chat prompt 字符串。"""
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
                txt = f"{self.system_prompt}\n\n{inst}\n\nAssistant:"
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
            self._graph_kv, g_c, g_d = self._encode_graphs(batch)

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

        self._graph_kv = None
        self.current_delta_h_g = None

        # 6. Plan-D 辅助 loss（cosine_only 或 two_phase）
        lm_loss  = outputs.loss
        aux_loss = torch.tensor(0.0, device=lm_loss.device)
        if 'label' in batch and g_c is not None:
            aux_loss = self._compute_aux_loss(g_c, g_d, batch['label'])

        aux_lambda = self._get_aux_lambda()
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
            self._graph_kv, _, _ = self._encode_graphs(batch)

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
        self.current_delta_h_g = None

        # 只截取新生成的 token
        new_ids = output_ids[:, enc.input_ids.shape[1]:]
        preds   = self.tokenizer.batch_decode(new_ids, skip_special_tokens=True)

        return {
            'id': batch['id'],
            'pred': preds,
            'label': batch['label'],
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
        log_rank0(f"可训练参数: {trainable:,} / 全部参数: {total:,} ({pct:.2f}%)")
        return trainable, total

    def remove_hook(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        if self._orig_attn_forward is not None:
            layers = self._get_transformer_layers()
            k = min(self.inject_layer, len(layers) - 1)
            self._get_self_attn(layers[k]).forward = self._orig_attn_forward
            self._orig_attn_forward = None
