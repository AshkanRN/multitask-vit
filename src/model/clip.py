from transformers import CLIPProcessor, CLIPVisionModel

processor = CLIPProcessor.from_pretrained("../models/clip-vit-base-patch16")
model = CLIPVisionModel.from_pretrained("../models/clip-vit-base-patch16")