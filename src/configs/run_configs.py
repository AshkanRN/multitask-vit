from dataclasses import dataclass, field, replace
from configs.adapter_config import AdapterConfig


@dataclass(frozen=True)
class SchedulerConfig:
    name: str = "warmup_cosine"

    warmup_epochs: int = 2
    warmup_start_factor: float = 0.1

    cosine_eta_min: float = 1e-6



@dataclass(frozen=True)
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 5
    min_delta: float = 0.001
    monitor: str = "weighted_f1"
    mode: str = "max"


@dataclass(frozen=True)
class RunConfig:
    age_loss_type: str = "ce"

    epochs: int = 20
    batch_size: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2

    age_dropout1: float = 0.15
    age_hidden_dim: int = 384

    backbone_fp32: bool = False

    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)



DEFAULT = RunConfig()


# Single source of truth for every (model_name, run_version) train.
# Add one entry here when you start a new run
RUN_CONFIGS = {
    ("clip",   "v1"):   replace(DEFAULT, age_loss_type="ce"),
    ("clip",   "v2"):   replace(DEFAULT, age_loss_type="ce"),

    ("dinov2", "v2"):   replace(DEFAULT, age_loss_type="ce"),
    ("siglip", "v2"):   replace(DEFAULT, age_loss_type="ce"),

    ("clip", "v3"): replace(
        DEFAULT,
        age_loss_type="corn",
        learning_rate=2e-4,
        scheduler=SchedulerConfig(
            warmup_epochs=3,
            cosine_eta_min=1e-6
        ),
        adapter=AdapterConfig(
            method="lora",
            r=8,
            alpha=16,
            dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        ),
        early_stopping = EarlyStoppingConfig(
            patience=7,
            min_delta=0.0005,
        )
    ),
    ("clip", "v3-test"): replace(
        DEFAULT,
        age_loss_type="corn",
        learning_rate=2e-4,
        scheduler=SchedulerConfig(
            warmup_epochs=3,
            cosine_eta_min=1e-6
        ),
        adapter=AdapterConfig(
            method="lora",
            r=8,
            alpha=16,
            dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        ),
        early_stopping = EarlyStoppingConfig(
            patience=7,
            min_delta=0.0005,
        )
    ),

    ("dinov2", "v3"): replace(
        DEFAULT,
        age_loss_type="corn",
        learning_rate=2e-4,
        backbone_fp32=True,
        scheduler=SchedulerConfig(
            warmup_epochs=3,
            cosine_eta_min=1e-6
        ),
        adapter=AdapterConfig(
            method="lora",
            r=8,
            alpha=16,
            dropout=0.05,
            target_modules=["query", "value"],
        ),
        early_stopping = EarlyStoppingConfig(
            patience=7,
            min_delta=0.0005,
        )
    ),
    ("siglip", "v3"): replace(
        DEFAULT,
        age_loss_type="corn",
        learning_rate=2e-4,
        scheduler=SchedulerConfig(
            warmup_epochs=3,
            cosine_eta_min=1e-6
        ),
        adapter=AdapterConfig(
            method="lora",
            r=8,
            alpha=16,
            dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        ),
        early_stopping = EarlyStoppingConfig(
            patience=7,
            min_delta=0.0005,
        )
    ),
    
}


def get_run_config(model_name, run_version):
    key = (model_name, run_version)
    if key not in RUN_CONFIGS:
        raise KeyError(
            f"No age_loss_type registered for {key}. "
            f"Add it to RUN_CONFIGS in configs/run_configs.py."
        )
    return RUN_CONFIGS[key]