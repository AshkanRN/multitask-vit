from transformers import AutoProcessor, AutoModel

processor = AutoProcessor.from_pretrained("../models/siglip-base-patch16-224")
model = AutoModel.from_pretrained("../models/siglip-base-patch16-224")