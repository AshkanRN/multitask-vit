# adapter.py

from peft import LoraConfig, get_peft_model, get_peft_model_state_dict


def apply_adapter(backbone, adapter_cfg):
    """
    Prepares the backbone for training according to adapter_cfg.
    "none": 
        fully frozen backbone (previous default behavior).
        
    "lora" or "dora" :
        backbone wrapped with PEFT, base weights frozen,
        only the LoRA/DoRA delta matrices are trainable.
    """
    if adapter_cfg.method == "none":
        for p in backbone.parameters():
            p.requires_grad = False
        return backbone


    if adapter_cfg.method in ("lora", "dora"):
        # for clip and Siglip: "k_proj", "v_proj", "q_proj"
        # for dinov2: "query", "value", "key"
        target_modules = adapter_cfg.target_modules or ["query", "value"]  

        peft_config = LoraConfig(
            r=adapter_cfg.r,
            lora_alpha=adapter_cfg.alpha,
            lora_dropout=adapter_cfg.dropout,
            target_modules=target_modules,
            use_dora=(adapter_cfg.method == "dora"),
        )
        return get_peft_model(backbone, peft_config)

    raise ValueError(f"Unknown adapter method: {adapter_cfg.method!r}")


def get_adapter_state(backbone, adapter_cfg):
    """Extracts only the small trainable delta weights, not the full backbone."""
    if adapter_cfg.method == "none":
        return None
    return get_peft_model_state_dict(backbone)