"""
Thin wrapper around vLLM's offline-batch LLM.chat() API.

Why this wrapper:
  - We want a *batched* interface: give me 64 chat conversations, get back
    64 parsed JSON dicts. Our prompts always ask for JSON.
  - We want robust JSON parsing (some models occasionally prepend a markdown
    fence even when told not to).
  - We want to hide chat-template details.

Usage:
    from shared.llm_batch import VLLMBatchClient

    client = VLLMBatchClient(
        model="Qwen/Qwen2.5-72B-Instruct-AWQ",
        tensor_parallel_size=2,
        max_model_len=8192,
        quantization="awq",
    )
    conversations = [
        [{"role": "system", "content": "..."},
         {"role": "user",   "content": "..."}],
        ...
    ]
    results = client.chat_json(conversations, temperature=0.0, max_tokens=512)
    # results is list[dict|None]; None means parse failure
"""

import json
import re
from typing import Any, Optional


_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def _extract_json(text: str) -> Optional[dict]:
    """
    Robustly extract a JSON object from an LLM response.
    Tolerates:
      - ```json ... ``` fences
      - leading/trailing whitespace or preamble before the object
    Returns None on failure.
    """
    if not text:
        return None
    # Strip markdown fence
    m = _JSON_FENCE_RE.match(text.strip())
    if m:
        text = m.group(1)
    # Fast path
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # Find the first balanced { ... } in the string
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


class VLLMBatchClient:
    """Lazy-loading, batched wrapper around vllm.LLM."""

    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 2,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.90,
        quantization: Optional[str] = None,
        dtype: str = "auto",
        seed: int = 42,
        trust_remote_code: bool = True,
    ):
        # Import lazily so unit tests / --help don't need vLLM installed.
        from vllm import LLM

        kwargs = dict(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            seed=seed,
            trust_remote_code=trust_remote_code,
        )
        if quantization:
            kwargs["quantization"] = quantization

        print(f"[vLLM] Loading {model} "
              f"(TP={tensor_parallel_size}, max_len={max_model_len}, "
              f"quant={quantization or 'none'})")
        self.llm = LLM(**kwargs)
        self.model_name = model
        print(f"[vLLM] Ready.")

    def chat_raw(
        self,
        conversations: list[list[dict]],
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 512,
        seed: Optional[int] = None,
    ) -> list[str]:
        """Return raw string outputs, one per conversation."""
        from vllm import SamplingParams

        sampling = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )
        outputs = self.llm.chat(conversations, sampling, use_tqdm=True)
        # vLLM returns them in the same order as input
        return [o.outputs[0].text if o.outputs else "" for o in outputs]

    def chat_json(
        self,
        conversations: list[list[dict]],
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 512,
        seed: Optional[int] = None,
    ) -> list[Optional[dict]]:
        """Return list of parsed JSON dicts; None for failures."""
        raw = self.chat_raw(
            conversations,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )
        return [_extract_json(r) for r in raw]


# ── Small convenience: build chat messages from system+user strings ──────────

def messages(system: str, user: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
