import os
import json

def load_text_file(filepath: str) -> str:
    """Reads and cleans text file content, ensuring UTF-8 encoding."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"提示词文件不存在: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read().strip()

def format_graph(graph_str: str) -> str:
    """
    Converts a JSON string of relation triples to readable text format.
    Input example: '[["A", "relation", "B"], ["C", "relation", "D"]]'
    Output example:
        - (A, relation, B)
        - (C, relation, D)
    Returns a fallback description if empty or failed to parse.
    """
    if not graph_str or not isinstance(graph_str, str):
        return "(no graph available)"
    try:
        triples = json.loads(graph_str)
        if not triples:
            return "(empty graph)"
        lines = [f"- ({t[0]}, {t[1]}, {t[2]})" for t in triples if len(t) == 3]
        return "\n".join(lines) if lines else "(empty graph)"
    except (json.JSONDecodeError, TypeError, IndexError):
        return graph_str.strip()

class BasePromptManager:
    """
    Base prompt manager that loads system_prompt.txt and user_prompt.txt
    from a specified directory.
    """
    def __init__(self, prompts_dir: str):
        # Resolve prompts_dir if it's relative
        from utils.path_utils import resolve_path
        resolved_prompts_dir = resolve_path(prompts_dir)
        
        system_prompt_path = os.path.join(resolved_prompts_dir, "system_prompt.txt")
        user_prompt_path = os.path.join(resolved_prompts_dir, "user_prompt.txt")
        
        self.system_prompt = load_text_file(system_prompt_path)
        self.user_prompt_template = load_text_file(user_prompt_path)

class PromptManager(BasePromptManager):
    """
    Prompt manager for knowledge graph extraction (graph_generate).
    Replaces {{content}} with the text to extract.
    """
    def get_messages(self, content: str) -> list:
        user_prompt = self.user_prompt_template.replace("{{content}}", content)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

class CoTPromptManager(BasePromptManager):
    """
    Prompt manager for Chain-of-Thought dataset generation (cot_generation).
    Replaces {{doc}}, {{claim}}, {{label}}, and {{label_desc}}.
    """
    def get_messages(self, doc: str, claim: str, label: int) -> list:
        label_desc = "Supported" if int(label) == 1 else "Unsupported"
        user_prompt = (
            self.user_prompt_template
            .replace("{{doc}}", str(doc))
            .replace("{{claim}}", str(claim))
            .replace("{{label}}", str(label))
            .replace("{{label_desc}}", label_desc)
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

class HalluPromptManager(BasePromptManager):
    """
    Prompt manager for standard hallucination detection (contrast_experiment).
    Replaces {{doc}} and {{claim}}.
    """
    def get_messages(self, doc: str, claim: str) -> list:
        user_prompt = (
            self.user_prompt_template
            .replace("{{doc}}", doc)
            .replace("{{claim}}", claim)
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

class AblationPromptManager(BasePromptManager):
    """
    Prompt manager for graph-augmented ablation experiment (ablation/kg).
    Replaces {{doc}}, {{claim}}, {{doc_graph}}, and {{claim_graph}}.
    """
    def get_messages(
        self,
        doc: str,
        claim: str,
        graph_doc: str = "",
        graph_claim: str = "",
    ) -> list:
        user_prompt = (
            self.user_prompt_template
            .replace("{{doc}}", doc)
            .replace("{{claim}}", claim)
            .replace("{{doc_graph}}", format_graph(graph_doc))
            .replace("{{claim_graph}}", format_graph(graph_claim))
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
