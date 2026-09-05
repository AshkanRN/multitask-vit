from torch import nn
from training.age_strategy import AGE_OUTPUT_DIMS
import torch

class MultiTaskViT(nn.Module):
    def __init__(self, backbone, dropout=0.15, age_hidden_dim=384, age_loss_type="ce"):
        super().__init__()

        # backbone is expected to already be frozen or PEFT-wrapped
        # by model.adapters.apply_adapter() before it's passed in here.
        self.backbone = backbone

        age_output_dim = AGE_OUTPUT_DIMS[age_loss_type]
        hidden_dim = 768

        self.gender = nn.Linear(hidden_dim, 1)  

        self.age = nn.Sequential(
            nn.Linear(hidden_dim, age_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(age_hidden_dim, age_output_dim)
        )

        self.race = nn.Linear(hidden_dim, 7)
        
    def forward(self, x):
        outputs = self.backbone(x)
        features = outputs.pooler_output

        gender_logits = self.gender(features)
        age_logits    = self.age(features)
        race_logits   = self.race(features)

        return  { 
            "gender": gender_logits, 
            "age":    age_logits, 
            "race":   race_logits
        }
