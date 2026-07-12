from transformers import AutoImageProcessor, AutoModel

processor = AutoImageProcessor.from_pretrained("../models/dinov2-base")
model = AutoModel.from_pretrained("../models/dinov2-base")