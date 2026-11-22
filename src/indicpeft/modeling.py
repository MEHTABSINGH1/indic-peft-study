"""Model construction for the three adaptation methods we compare.

* ``full``  - every weight is trained (the upper-bound baseline).
* ``lora``  - frozen backbone + low-rank adapters (Hu et al., 2021).
* ``qlora`` - 4-bit NF4 frozen backbone + LoRA adapters (Dettmers et al., 2023).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

from .data import LABELS

# Attention projections per architecture family. "all-linear" is also accepted.
TARGETS = {
    "attn_qv": {"bert": ["query", "value"], "xlm-roberta": ["query", "value"],
                "gemma": ["q_proj", "v_proj"], "gemma2": ["q_proj", "v_proj"],
                "llama": ["q_proj", "v_proj"], "qwen2": ["q_proj", "v_proj"]},
    "attn_all": {"bert": ["query", "key", "value", "attention.output.dense"],
                 "xlm-roberta": ["query", "key", "value", "attention.output.dense"],
                 "gemma": ["q_proj", "k_proj", "v_proj", "o_proj"],
                 "gemma2": ["q_proj", "k_proj", "v_proj", "o_proj"],
                 "llama": ["q_proj", "k_proj", "v_proj", "o_proj"],
                 "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj"]},
}


@dataclass
class ParamCount:
    trainable: int
    total: int

    @property
    def pct(self) -> float:
        return 100.0 * self.trainable / max(self.total, 1)


def count_params(model: torch.nn.Module) -> ParamCount:
    trainable = total = 0
    for p in model.parameters():
        n = p.numel()
        # 4-bit params are packed two-per-byte; report logical size.
        if p.__class__.__name__ == "Params4bit":
            n *= 2
        total += n
        if p.requires_grad:
            trainable += n
    return ParamCount(trainable, total)


def _resolve_targets(model, targets: str) -> list[str] | str:
    if targets == "all-linear":
        return "all-linear"
    family = model.config.model_type
    try:
        return TARGETS[targets][family]
    except KeyError as e:
        raise ValueError(f"No '{targets}' target list for model_type={family}") from e


DECODERS = {"gemma", "gemma2", "gemma3", "llama", "qwen2", "mistral"}


class PooledClassifier(nn.Module):
    """Encoder + pooling + linear head.

    We do NOT use the checkpoint's [CLS] pooler. For MuRIL its output is almost
    identical for every sentence (mean pairwise cosine 0.999, per-dimension std
    ~3e-4 on SIB-200 Punjabi), so a freshly initialised head cannot separate
    classes and collapses to the majority class. Mean-pooling the token states
    gives well-scaled, discriminative features (frozen linear probe: 0.78 vs.
    0.74 macro-F1 with standardised pooler features). Decoder LLMs use the
    last non-padding token instead, since only it has attended to the whole
    sentence.
    """

    def __init__(self, encoder: nn.Module, num_labels: int, pooling: str, dropout: float = 0.1):
        super().__init__()
        self.encoder, self.pooling = encoder, pooling
        self.config = encoder.config
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder.config.hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **_):
        extra = {} if token_type_ids is None else {"token_type_ids": token_type_ids}
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **extra).last_hidden_state
        if self.pooling == "mean":
            m = attention_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        else:  # last non-padding token (right padding)
            last = attention_mask.sum(1) - 1
            pooled = h[torch.arange(h.size(0), device=h.device), last]
        logits = self.classifier(self.dropout(pooled.float()))
        loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)


def build(model_name: str, method: str, lora_r: int = 8, lora_alpha: int | None = None,
          lora_dropout: float = 0.1, lora_targets: str = "attn_qv"):
    """Return (model, tokenizer) ready for training with the chosen method."""
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    kwargs: dict = {}
    if method == "qlora":
        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA needs a CUDA GPU (bitsandbytes 4-bit kernels).")
        from transformers import BitsAndBytesConfig
        compute = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute)
        # `torch_dtype` was renamed to `dtype` in transformers 4.56.
        import transformers
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        kwargs["dtype" if (major, minor) >= (4, 56) else "torch_dtype"] = compute
    if method not in {"full", "lora", "qlora"}:
        raise ValueError(f"Unknown method '{method}'")
    encoder = AutoModel.from_pretrained(model_name, **kwargs)
    pooling = "last" if encoder.config.model_type in DECODERS else "mean"

    if method != "full":
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        if method == "qlora":
            encoder = prepare_model_for_kbit_training(encoder, use_gradient_checkpointing=True)
        cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, r=lora_r, lora_alpha=lora_alpha or 2 * lora_r,
            lora_dropout=lora_dropout, target_modules=_resolve_targets(encoder, lora_targets))
        encoder = get_peft_model(encoder, cfg)  # freezes the backbone, adds adapters
    # The head sits outside the PEFT wrapper, so it is always fully trainable.
    return PooledClassifier(encoder, len(LABELS), pooling), tok
