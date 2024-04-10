from transformers import ViltProcessor, ViltForQuestionAnswering

model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
import requests
from PIL import Image
import numpy as np

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# download an input image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
rgb_image = image.convert('RGB')
# print('image shape =', image_np.shape)
text = "How many cats are there?"
text = "What do you see in this image?"

# prepare inputs
inputs = processor(rgb_image, text, return_tensors="pt")
import torch

# forward pass
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
idx = logits.argmax(-1).item()
print("Predicted answer:", model.config.id2label[idx])
