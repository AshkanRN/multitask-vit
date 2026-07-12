from transformers import Sam3Model, Sam3Processor
import torch
model = Sam3Model.from_pretrained("../models/facebook--sam3")
model.eval()

processor = Sam3Processor.from_pretrained("../models/facebook--sam3")