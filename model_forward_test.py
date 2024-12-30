import requests
from PIL import Image

from bge_gemma2_multimodal import BgeGemma2MultimodalProcessor, BgeGemma2MultimodalConfig, BgeGemma2MultimodalModel

config = {
    "image_token_index":    255999,
    "model_type":           "bge_gemma2_multimodal",
    "projection_dim":       64,
    "text_config":          {
        "_name_or_path":         "BAAI/bge-multilingual-gemma2",
        "architectures":         [
            "Gemma2Model"
            ],
        "hidden_act":            "gelu_pytorch_tanh",
        "hidden_size":           64,
        "intermediate_size":     64,
        "model_type":            "gemma2",
        "num_attention_heads":   2,
        "num_hidden_layers":     2,
        "num_image_tokens":      196,
        "num_key_value_heads":   2,
        "query_pre_attn_scalar": 2,
        "sliding_window":        128,
        "torch_dtype":           "bfloat16",
        "use_cache":             False,
        "vocab_size":            256002
        },
    "text_pretrained":      "BAAI/bge-multilingual-gemma2",
    "transformers_version": "4.47.0",
    "vision_config":        {
        "model_type":       "siglip_vision_model",
        "num_image_tokens": 196,
        "num_positions":    196,
        "torch_dtype":      "bfloat16",
        "projection_dim":   64,
        "vision_use_head":  False,
        },
    "vision_pretrained":    "google/siglip-base-patch16-224"
    }
# Creat model config
config = BgeGemma2MultimodalConfig.from_dict(config)
model = BgeGemma2MultimodalModel(config)
processor = BgeGemma2MultimodalProcessor.from_pretrained("bge_gemma2_multimodal_hub_files")

# Define input image and texts
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
texts = ["a photo of 2 cats", ]
instruct = "Multimodal search retrieval"

# *** Image & Text Index ***
inputs = processor(text=texts,
                   images=image,
                   padding="max_length",
                   return_tensors="pt")
outputs = model(**inputs)

# *** Image & Text Query ***
inputs = processor(text=texts,
                   images=image,
                   instruct=instruct,
                   padding="max_length",
                   return_tensors="pt")
outputs = model(**inputs)

# *** Image Index ***
inputs = processor(text=None,
                   images=image,
                   padding="max_length",
                   return_tensors="pt")
outputs = model(**inputs)

# *** Image Query ***
inputs = processor(text=None,
                   images=image,
                   instruct=instruct,
                   padding="max_length",
                   return_tensors="pt")
outputs = model(**inputs)

# *** Text Index ***
inputs = processor(text=texts,
                   images=None,
                   padding="max_length",
                   return_tensors="pt")
outputs = model(**inputs)

# *** Text Query ***
inputs = processor(text=texts,
                   images=None,
                   instruct=instruct,
                   padding="max_length",
                   return_tensors="pt")
outputs = model(**inputs)
