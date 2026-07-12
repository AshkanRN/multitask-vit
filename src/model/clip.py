from transformers import AutoProcessor, AutoModel

processor = AutoProcessor.from_pretrained("../models/clip-vit-base-patch16")
model = AutoModel.from_pretrained("../models/clip-vit-base-patch16")