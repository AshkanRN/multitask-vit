from torch import nn
from training.age_strategies import AGE_OUTPUT_DIMS

class FairFaceViT(nn.Module):
    def __init__(self, model, dropout=0.15, age_hidden_dim=384, age_loss_type="ce"):
        super().__init__()

        self.backbone = model

        for param in self.backbone.parameters():
            param.requires_grad = False


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
