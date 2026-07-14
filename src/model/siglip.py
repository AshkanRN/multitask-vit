# from transformers import AutoProcessor, AutoModel
from transformers import SiglipVisionModel, SiglipImageProcessor

processor = SiglipImageProcessor.from_pretrained("../models/siglip-base-patch16-224")
model = SiglipVisionModel.from_pretrained("../models/siglip-base-patch16-224")