from torch import nn


class FairFaceViT(nn.Module):
    def __init__(self, model, dropout=0.15, age_hidden_dim=384):
        super().__init__()

        self.backbone = model

        print('backbone: ',self.backbone,"\n")
        print('backbone parameters: ',self.backbone.parameters(),"\n")

        print("start freezing the backbone parameters...\n")

        # Freeze the backbone parameters
        for param in self.backbone.parameters():
            print(param)
            param.requires_grad = False

        print("finished freezing the backbone parameters...\n")

        hidden_dim = 768

        self.gender = nn.Linear(hidden_dim, 1)
            
        self.age = nn.Sequential(
            nn.Linear(hidden_dim, age_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(age_hidden_dim, 9)
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