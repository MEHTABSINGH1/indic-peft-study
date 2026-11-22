"""Model construction for the three adaptation methods we compare.

* ``full``  - every weight is trained (the upper-bound baseline).
* ``lora``  - frozen backbone + low-rank adapters (Hu et al., 2021).
* ``qlora`` - 4-bit NF4 frozen backbone + LoRA adapters (Dettmers et al., 2023).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import ID2LABEL, LABEL2ID

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


def build(model_name: str, method: str, lora_r: int = 8, lora_alpha: int | None = None,
          lora_dropout: float = 0.1, lora_targets: str = "attn_qv"):
    """Return (model, tokenizer) ready for training with the chosen method."""
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ignore_mismatched_sizes: re-initialise any existing classification head
    # (needed when a checkpoint ships with a head for a different label count).
    kwargs = dict(num_labels=len(ID2LABEL), id2label=ID2LABEL, label2id=LABEL2ID,
                  ignore_mismatched_sizes=True)
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
    model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
    model.config.pad_token_id = tok.pad_token_id

    if method == "full":
        return model, tok
    if method not in {"lora", "qlora"}:
        raise ValueError(f"Unknown method '{method}'")

    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    if method == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=lora_r, lora_alpha=lora_alpha or 2 * lora_r,
        lora_dropout=lora_dropout, target_modules=_resolve_targets(model, lora_targets))
    # TaskType.SEQ_CLS keeps the new classification head trainable.
    return get_peft_model(model, cfg), tok
