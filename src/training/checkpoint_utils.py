#checkpoint_utils.py

from configs.adapter_config import AdapterConfig
from model.adapters import apply_adapter
from fairface_vit import FairFaceViT

try:
    from peft import set_peft_model_state_dict
except ImportError:
    set_peft_model_state_dict = None


def build_model_from_checkpoint(checkpoint, raw_backbone, device="cpu"):
    """
    Reconstructs a FairFaceViT from a checkpoint dict, self-configuring:
      - age_loss_type      (defaults to "ce" for old checkpoints)
      - adapter method      (defaults to "none" for old checkpoints -> frozen backbone)
      - head architecture   (age_dropout1 / age_hidden_dim, read back from the checkpoint)
    Works unchanged for checkpoints saved before LoRA/DoRA existed.
    """
    adapter_cfg = AdapterConfig(**(checkpoint.get("adapter_config") or {"method": "none"}))
    age_loss_type = checkpoint.get("age_loss_type", "ce")
    age_dropout1 = checkpoint.get("age_dropout1", 0.15)
    age_hidden_dim = checkpoint.get("age_hidden_dim", 384)

    backbone = apply_adapter(raw_backbone, adapter_cfg)

    model = FairFaceViT(
        backbone,
        dropout=age_dropout1,
        age_hidden_dim=age_hidden_dim,
        age_loss_type=age_loss_type,
    )

    model.gender.load_state_dict(checkpoint["gender_head"])
    model.age.load_state_dict(checkpoint["age_head"])
    model.race.load_state_dict(checkpoint["race_head"])

    if adapter_cfg.method != "none":
        if set_peft_model_state_dict is None:
            raise ImportError("peft is required to load an adapter checkpoint: pip install peft")
        set_peft_model_state_dict(model.backbone, checkpoint["adapter_state_dict"])

    model.to(device)
    model.eval()

    return model, age_loss_type, adapter_cfg