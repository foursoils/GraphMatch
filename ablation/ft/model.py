"""Qwen3-4B + LoRA 纯文本幻觉检测模型。"""

import contextlib
import os

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.path_utils import configure_dist_process_logging, log_rank0, resolve_path

try:
    from peft import LoraConfig, TaskType, get_peft_model
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False

IGNORE_INDEX = -100


class LoraHalluModel(nn.Module):
    """纯文本 Causal LM + LoRA；forward 返回 loss，inference 返回生成文本。"""

    def __init__(self, config: dict, device: torch.device = None, apply_lora: bool = True):
        super().__init__()
        model_cfg = config['model']
        lora_cfg = config['lora']

        self.max_txt_len = model_cfg.get('max_txt_len', 2048)
        self.max_new_tokens = model_cfg.get('max_new_tokens', 2)
        self.system_prompt = None  # 由 batch 或外部 prompt 注入

        llm_path = resolve_path(model_cfg['llm_model_path'])

        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=True, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'left'

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
                device_map = {"": 0} if torch.cuda.is_available() else None
                _dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        log_rank0(f"[Init] 加载 LLM: {llm_path} (device_map={device_map})")
        configure_dist_process_logging()
        llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            dtype=torch.bfloat16,
            attn_implementation=model_cfg.get('attn_implementation', 'sdpa'),
            low_cpu_mem_usage=True,
            device_map=device_map,
            trust_remote_code=True,
        )

        if _PEFT_AVAILABLE and apply_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_cfg.get('r', 8),
                lora_alpha=lora_cfg.get('lora_alpha', 16),
                lora_dropout=lora_cfg.get('lora_dropout', 0.05),
                target_modules=lora_cfg.get(
                    'target_modules', ['q_proj', 'k_proj', 'v_proj', 'o_proj']
                ),
                bias='none',
            )
            llm = get_peft_model(llm, lora_config)
            log_rank0("[Init] LoRA 已应用到全部目标模块。")
        elif not apply_lora:
            log_rank0("[Init] 推理模式：跳过 LoRA 初始化，等待加载 adapter。")
        else:
            log_rank0("[Init] 无 peft，LLM 全量可训练（仅调试）。")

        self.llm = llm
        if _dev.type == 'cuda':
            self.device_id = _dev.index if _dev.index is not None else 0
        else:
            self.device_id = 0

        log_rank0("[Init] LoraHalluModel 就绪。")

    def _device(self) -> torch.device:
        return torch.device(f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu")

    def maybe_autocast(self, dtype=torch.bfloat16):
        if torch.cuda.is_available():
            return torch.amp.autocast(device_type='cuda', dtype=dtype)
        return contextlib.nullcontext()

    def _make_chat_messages(self, system_prompt: str, instruction: str):
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
        ]

    def _apply_chat_template(self, system_prompt: str, instructions: list, add_generation_prompt: bool = True):
        prompts = []
        for inst in instructions:
            msgs = self._make_chat_messages(system_prompt, inst)
            try:
                txt = self.tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            except Exception:
                txt = f"{system_prompt}\n\n{inst}\n\nAssistant:"
            prompts.append(txt)
        return prompts

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

    def forward(self, batch: dict) -> torch.Tensor:
        """SFT：仅对 target（0/1）计算 loss。"""
        _dev = self._device()
        system_prompt = batch.get('system_prompt') or self.system_prompt or ""
        prompts = self._apply_chat_template(system_prompt, batch['instruction'], add_generation_prompt=True)
        targets = batch['target']
        full_texts = [p + t for p, t in zip(prompts, targets)]

        enc_full = self.tokenizer(
            full_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_txt_len + self.max_new_tokens,
        ).to(_dev)

        enc_prompt = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_txt_len,
        ).to(_dev)

        input_ids = enc_full.input_ids
        attention_mask = enc_full.attention_mask
        labels = input_ids.clone()
        seq_len = input_ids.shape[1]
        full_lens = enc_full.attention_mask.sum(dim=1).tolist()
        prompt_lens = enc_prompt.attention_mask.sum(dim=1).tolist()
        for i, (full_len, prompt_len) in enumerate(zip(full_lens, prompt_lens)):
            prompt_end = seq_len - int(full_len) + int(prompt_len)
            labels[i, :prompt_end] = IGNORE_INDEX
        labels[attention_mask == 0] = IGNORE_INDEX

        with self.maybe_autocast():
            outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
        return outputs.loss

    @torch.no_grad()
    def inference(self, batch: dict) -> dict:
        _dev = self._device()
        system_prompt = batch.get('system_prompt') or self.system_prompt or ""
        prompts = self._apply_chat_template(system_prompt, batch['instruction'], add_generation_prompt=True)
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

        new_ids = output_ids[:, enc.input_ids.shape[1]:]
        preds = self.tokenizer.batch_decode(new_ids, skip_special_tokens=True)
        return {
            'id': batch['id'],
            'pred': preds,
            'label': batch['label'],
        }
