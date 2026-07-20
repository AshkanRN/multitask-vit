from transformers import CLIPProcessor, CLIPVisionModel

processor = CLIPProcessor.from_pretrained("../models/clip-vit-base-patch16").image_processor
model = CLIPVisionModel.from_pretrained("../models/clip-vit-base-patch16")