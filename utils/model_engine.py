import re
import json
import concurrent.futures
from typing import List, Union, Tuple

# ---------------------------------------------------------------------------
# Centralized Utilities
# ---------------------------------------------------------------------------

def parse_binary_label(raw: str) -> int:
    """
    Extracts 0 or 1 from the raw output string of the model.
    Strategy: Matches the last occurrence of word-bounded 0 or 1.
    Returns 2 (invalid/parsing failure) if no match is found or input is empty.
    """
    if not raw:
        return 2
    text = raw.strip()
    matches = re.findall(r'\b([01])\b', text)
    if matches:
        return int(matches[-1])
    return 2

# ---------------------------------------------------------------------------
# Unified Inference Engines
# ---------------------------------------------------------------------------

class VLLMEngine:
    """Uses the local offline vLLM engine for batch inference."""
    
    def __init__(self, config: dict):
        from vllm import LLM, SamplingParams
        from utils.path_utils import resolve_path
        
        # Support both flat and nested configuration formats
        vllm_cfg = config.get('inference', {}).get('vllm', {})
        model_path = vllm_cfg.get('vllm_model_path') or config.get('inference', {}).get('vllm_model_path')
        if not model_path:
            raise ValueError("vLLM model path must be configured in 'inference.vllm_model_path' or 'inference.vllm.vllm_model_path'")
            
        model_path = resolve_path(model_path)
        
        gpu_mem = vllm_cfg.get('vllm_gpu_memory_utilization') or config.get('inference', {}).get('vllm_gpu_memory_utilization', 0.9)
        tp_size = vllm_cfg.get('vllm_tensor_parallel_size') or config.get('inference', {}).get('vllm_tensor_parallel_size', 1)
        max_model_len = vllm_cfg.get('vllm_max_model_len') or config.get('inference', {}).get('vllm_max_model_len', None)
        max_num_seqs = vllm_cfg.get('vllm_max_num_seqs') or config.get('inference', {}).get('vllm_max_num_seqs', None)
        if max_num_seqs is not None:
            max_num_seqs = int(max_num_seqs)
        enable_chunked_prefill = vllm_cfg.get('vllm_enable_chunked_prefill') or config.get('inference', {}).get('vllm_enable_chunked_prefill', False)
        
        print(f"[vLLM] 正在加载模型: {model_path} (max_num_seqs={max_num_seqs}, gpu_mem={gpu_mem}, tp_size={tp_size})")
        
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_mem,
            tensor_parallel_size=tp_size,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            enable_chunked_prefill=enable_chunked_prefill
        )
        
        model_cfg = config.get('model', {})
        self.sampling_params = SamplingParams(
            max_tokens=model_cfg.get('max_new_tokens', 16),
            temperature=model_cfg.get('temperature', 0.0),
            top_p=model_cfg.get('top_p', 1.0),
            top_k=model_cfg.get('top_k', -1),
            min_p=model_cfg.get('min_p', 0.0),
            presence_penalty=model_cfg.get('presence_penalty', 0.0),
            repetition_penalty=model_cfg.get('repetition_penalty', 1.0),
        )
        
        self.enable_thinking = model_cfg.get('enable_thinking', False)
        
    def batch_infer(self, messages_list: List[List[dict]]) -> List[str]:
        """
        Runs batch inference.
        messages_list: List of chat message lists (OpenAI role-content format)
        Returns: List of raw text output strings
        """
        outputs = self.llm.chat(
            messages=messages_list,
            sampling_params=self.sampling_params,
            chat_template_kwargs={"enable_thinking": self.enable_thinking},
            use_tqdm=True,
        )
        return [out.outputs[0].text for out in outputs]


class APIEngine:
    """Runs concurrent inference using an OpenAI-compatible API endpoint."""
    
    def __init__(self, config: dict):
        from openai import OpenAI
        
        api_cfg = config.get('inference', {}).get('api', {})
        api_url = api_cfg.get('api_base_url') or config.get('inference', {}).get('api_base_url', 'http://localhost:8000/v1')
        api_key = api_cfg.get('api_key') or config.get('inference', {}).get('api_key', 'EMPTY')
        
        self.client = OpenAI(base_url=api_url, api_key=api_key)
        
        # Determine model name
        configured_model_name = api_cfg.get('api_model_name') or config.get('inference', {}).get('api_model_name')
        if configured_model_name:
            self.model_name = configured_model_name
        else:
            self.model_name = self.client.models.list().data[0].id
            
        model_cfg = config.get('model', {})
        self.max_tokens = model_cfg.get('max_new_tokens', 16)
        self.temperature = model_cfg.get('temperature', 0.0)
        self.top_p = model_cfg.get('top_p', 1.0)
        self.top_k = model_cfg.get('top_k')
        self.min_p = model_cfg.get('min_p')
        self.presence_penalty = model_cfg.get('presence_penalty')
        self.repetition_penalty = model_cfg.get('repetition_penalty')
        self.enable_thinking = model_cfg.get('enable_thinking', False)
        
        self.max_workers = api_cfg.get('api_max_workers') or config.get('inference', {}).get('api_max_workers', 8)
        
    def _single_infer(self, messages: List[dict]) -> str:
        extra_body = {"enable_thinking": self.enable_thinking}
        if self.top_k is not None:
            extra_body["top_k"] = self.top_k
        if self.min_p is not None:
            extra_body["min_p"] = self.min_p
        if self.repetition_penalty is not None:
            extra_body["repetition_penalty"] = self.repetition_penalty
            
        params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "extra_body": extra_body
        }
        if self.presence_penalty is not None:
            params["presence_penalty"] = self.presence_penalty
            
        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content or ""
        
    def batch_infer(self, messages_list: List[List[dict]], pbar=None) -> List[str]:
        """Runs concurrent API inference."""
        results = [""] * len(messages_list)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self._single_infer, msgs): i
                for i, msgs in enumerate(messages_list)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"[API Error] index={idx}: {e}")
                    results[idx] = "FAILED"
                if pbar is not None:
                    pbar.update(1)
        return results


class NLIEngine:
    """
    Runs sequence classification models (BERT/DeBERTa/RoBERTa) for NLI verification.
    """
    
    def __init__(self, config: dict):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from utils.path_utils import resolve_path
        
        model_path = config.get('inference', {}).get('nli_model_path')
        if not model_path:
            raise ValueError("NLI model path must be configured in 'inference.nli_model_path'")
            
        model_path = resolve_path(model_path)
        print(f"[NLI] 加载模型: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        print(f"[NLI] 模型已加载到: {self.device}")
        
        self.max_length = config.get('inference', {}).get('nli_max_length', 512)
        
        # Detect entailment class ID from model config's id2label
        _entail_keywords = {"entailment", "supported", "support", "yes", "true"}
        id2label = getattr(self.model.config, 'id2label', {})
        auto_id = None
        for idx, lbl in id2label.items():
            if str(lbl).lower().strip() in _entail_keywords:
                auto_id = int(idx)
                break
                
        manual_id = config.get('inference', {}).get('nli_entailment_id', None)
        if manual_id is not None:
            self.entailment_id = int(manual_id)
            print(f"[NLI] 使用手动配置的 entailment_id={self.entailment_id}")
        elif auto_id is not None:
            self.entailment_id = auto_id
            print(f"[NLI] 自动检测 entailment_id={self.entailment_id} (label='{id2label[auto_id]}')")
        else:
            self.entailment_id = 1
            print(f"[NLI] 未能自动检测，使用默认 entailment_id={self.entailment_id}，id2label={id2label}")
            
        self._torch = torch
        
    def batch_infer(self, inputs: List[Tuple[str, str]]) -> List[str]:
        """
        inputs: List of (doc, claim) tuples
        Returns: List of "0" or "1" strings
        """
        docs = [item[0] for item in inputs]
        claims = [item[1] for item in inputs]
        
        encoded = self.tokenizer(
            docs,
            claims,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        ).to(self.device)
        
        with self._torch.no_grad():
            logits = self.model(**encoded).logits
            
        preds = logits.argmax(dim=-1).cpu().tolist()
        return ["1" if p == self.entailment_id else "0" for p in preds]


def build_engine(config: dict) -> Union[VLLMEngine, APIEngine, NLIEngine]:
    """Unified factory function to build model engines based on engine_type."""
    engine_type = config.get('inference', {}).get('engine_type', 'vllm').lower()
    if engine_type == 'vllm':
        return VLLMEngine(config)
    elif engine_type == 'api':
        return APIEngine(config)
    elif engine_type == 'nli':
        return NLIEngine(config)
    else:
        raise ValueError(f"Unsupported engine_type: '{engine_type}'. Choose 'vllm', 'api', or 'nli'.")

# ---------------------------------------------------------------------------
# Specialized Knowledge Graph Extraction Engine
# ---------------------------------------------------------------------------

class LocalQwenExtractor:
    """
    Dedicated Knowledge Graph extraction engine wrapper.
    Decoupled and structured to match local/api extraction workflows.
    """
    def __init__(self, config: dict):
        engine_type = config.get('inference', {}).get('engine_type', 'api').lower()
        if engine_type == 'api':
            self.engine = APIEngine(config)
            # Expose attributes directly for backwards compatibility with retry_main.py and extraction_main.py
            self.client = self.engine.client
            self.serving_model = self.engine.model_name
            self.max_tokens = self.engine.max_tokens
            self.temperature = self.engine.temperature
            self.top_p = self.engine.top_p
            self.presence_penalty = self.engine.presence_penalty
            self.top_k = self.engine.top_k
            self.min_p = self.engine.min_p
            self.repetition_penalty = self.engine.repetition_penalty
            self.enable_thinking = self.engine.enable_thinking
            self.api_max_workers = self.engine.max_workers
        elif engine_type == 'vllm':
            self.engine = VLLMEngine(config)
        else:
            raise ValueError(f"Extractor does not support engine type: {engine_type}")
            
        self.engine_type = engine_type
        
    def batch_extract(self, messages_list: list, pbar=None) -> list:
        if self.engine_type == 'api':
            return self.engine.batch_infer(messages_list, pbar=pbar)
        else:
            return self.engine.batch_infer(messages_list)
            
    def _parse_graph_tag(self, response: str) -> tuple:
        """
        Parses knowledge graph triples enclosed in <graph>...</graph> tags.
        Returns: (status, parsed_triplets) where status=1 on success, 0 on failure.
        """
        if not response or response == "FAILED":
             return 0, []
             
        end_idx = response.rfind('</graph>')
        if end_idx != -1:
            start_idx = response.rfind('<graph>', 0, end_idx)
            if start_idx != -1:
                raw_str = response[start_idx + 7:end_idx].strip()
            else:
                raw_str = response.strip()
        else:
            raw_str = response.strip()
            
        try:
            parsed_data = json.loads(raw_str)
            if not isinstance(parsed_data, list) or len(parsed_data) == 0:
                 return 0, []
            for item in parsed_data:
                 if not isinstance(item, list) or len(item) != 3:
                     return 0, []
            return 1, parsed_data
        except json.JSONDecodeError:
            return 0, []
