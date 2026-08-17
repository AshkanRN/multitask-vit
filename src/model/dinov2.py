from transformers import AutoImageProcessor, AutoModel

processor = AutoImageProcessor.from_pretrained("../models/dinov2-base")

model = AutoModel.from_pretrained(
    "../models/dinov2-base",
    attn_implementation="eager"
)

# processor = AutoImageProcessor.from_pretrained("/home/ashkanrn/01-Project/University/Deep-Learning/DL-project/models/dinov2-base")
# model = AutoModel.from_pretrained("/home/ashkanrn/01-Project/University/Deep-Learning/DL-project/models/dinov2-base")


# print(type(processor))
# print(processor)