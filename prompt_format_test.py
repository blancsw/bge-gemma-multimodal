import requests
from PIL import Image

from bge_gemma2_multimodal import BgeGemma2MultimodalProcessor

# Load the multimodal processor
processor = BgeGemma2MultimodalProcessor.from_pretrained("bge_gemma2_multimodal_hub_files")

# Define input image and texts
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
texts = ["a photo of 2 cats", ]
instruct = "Multimodal search retrieval"

# *** Image & Text Index ***
inputs = processor(text=texts,
                   images=image,
                   return_formated_prompt=True,
                   padding="max_length",
                   return_tensors="pt")
print("Formatted prompt for image & text index:")
print(inputs.formated_prompt)

# *** Image & Text Query ***
inputs = processor(text=texts,
                   images=image,
                   return_formated_prompt=True,
                   instruct=instruct,
                   padding="max_length",
                   return_tensors="pt")
print("Formatted prompt for image & text query:")
print(inputs.formated_prompt)

# *** Image Index ***
inputs = processor(text=None,
                   images=image,
                   return_formated_prompt=True,
                   padding="max_length",
                   return_tensors="pt")
print("Formatted prompt for image index:")
print(inputs.formated_prompt)

# *** Image Query ***
inputs = processor(text=None,
                   images=image,
                   return_formated_prompt=True,
                   instruct=instruct,
                   padding="max_length",
                   return_tensors="pt")
print("Formatted prompt for image query:")
print(inputs.formated_prompt)

# *** Text Index ***
inputs = processor(text=texts,
                   images=None,
                   return_formated_prompt=True,
                   padding="max_length",
                   return_tensors="pt")
print("Formatted prompt for text index:")
print(inputs.formated_prompt)

# *** Text Query ***
inputs = processor(text=texts,
                   images=None,
                   return_formated_prompt=True,
                   instruct=instruct,
                   padding="max_length",
                   return_tensors="pt")
print("Formatted prompt for text query:")
print(inputs.formated_prompt)
