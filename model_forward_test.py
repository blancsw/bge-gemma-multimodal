import requests
from PIL import Image
from transformers import SiglipVisionConfig
from transformers.models.gemma2 import Gemma2Config

from bge_gemma2_multimodal import BgeGemma2MultimodalProcessor, BgeGemma2MultimodalConfig, BgeGemma2MultimodalModel

text_config = Gemma2Config.from_dict({
    "attention_bias":          False,
    "attention_dropout":       0.0,
    "attn_logit_softcapping":  50.0,
    "bos_token_id":            2,
    "cache_implementation":    "hybrid",
    "eos_token_id":            1,
    "final_logit_softcapping": 30.0,
    "head_dim":                64,
    "hidden_act":              "gelu_pytorch_tanh",
    "hidden_activation":       "gelu_pytorch_tanh",
    "hidden_size":             64,
    "initializer_range":       0.02,
    "intermediate_size":       64,
    "max_position_embeddings": 1024,
    "model_type":              "gemma2",
    "num_attention_heads":     2,
    "num_hidden_layers":       2,
    "num_key_value_heads":     2,
    "pad_token_id":            0,
    "query_pre_attn_scalar":   64,
    "rms_norm_eps":            1e-06,
    "rope_theta":              10000.0,
    "sliding_window":          256,
    "sliding_window_size":     256,
    "use_cache":               False,
    "vocab_size":              256002
    })
vision_config = SiglipVisionConfig.from_pretrained("google/siglip-base-patch16-224")

# Creat model config
config = BgeGemma2MultimodalConfig(vision_config=vision_config,
                                   projection_dim=64,
                                   text_config=text_config)
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

# # *** Image & Text Query ***
# inputs = processor(text=texts,
#                    images=image,
#                    instruct=instruct,
#                    padding="max_length",
#                    return_tensors="pt")
# print("Formatted prompt for image & text query:")
# print(inputs.formated_prompt)
#
# # *** Image Index ***
# inputs = processor(text=None,
#                    images=image,
#                    padding="max_length",
#                    return_tensors="pt")
# print("Formatted prompt for image index:")
# print(inputs.formated_prompt)
#
# # *** Image Query ***
# inputs = processor(text=None,
#                    images=image,
#                    instruct=instruct,
#                    padding="max_length",
#                    return_tensors="pt")
# print("Formatted prompt for image query:")
# print(inputs.formated_prompt)
#
# # *** Text Index ***
# inputs = processor(text=texts,
#                    images=None,
#                    padding="max_length",
#                    return_tensors="pt")
# print("Formatted prompt for text index:")
# print(inputs.formated_prompt)
#
# # *** Text Query ***
# inputs = processor(text=texts,
#                    images=None,
#                    instruct=instruct,
#                    padding="max_length",
#                    return_tensors="pt")
# print("Formatted prompt for text query:")
# print(inputs.formated_prompt)
