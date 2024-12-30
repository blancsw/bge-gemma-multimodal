"""BGE Multimodal configuration"""
from typing import Union

from transformers import PretrainedConfig, CONFIG_MAPPING
from transformers.models.gemma2 import Gemma2Config
from transformers.models.siglip import SiglipVisionConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class BgeGemma2MultimodalConfig(PretrainedConfig):

    model_type = "bge_gemma2_multimodal"
    sub_configs = {"text_config": Gemma2Config, "vision_config": SiglipVisionConfig}

    def __init__(
            self,
            text_config: Union[Gemma2Config, dict] = None,
            vision_config: Union[SiglipVisionConfig, dict] = None,
            text_pretrained: str = "BAAI/bge-multilingual-gemma2",
            vision_pretrained: str = "google/siglip-base-patch16-224",
            projection_dim=2048,
            image_token_index=255999,
            **kwargs
            ):
        """

        Args:
            text_config:
            vision_config:
            text_pretrained:
            vision_pretrained:
            projection_dim:
            image_token_index: index of the token <vision> see tokenizer_config.json for index value
            **kwargs:
        """
        self.text_pretrained = text_pretrained
        self.vision_pretrained = vision_pretrained
        self.image_token_index = image_token_index
        # If config objects are passed directly, use them; otherwise, instantiate from dict
        if isinstance(text_config, Gemma2Config):
            self.text_config = text_config
        elif isinstance(text_config, dict):
            self.text_config = Gemma2Config(**(text_config or {}))
        else:
            assert text_pretrained is not None, f"You must provide text_config or text_pretrained parameter"
            self.text_config = Gemma2Config.from_pretrained(text_pretrained)

        if isinstance(vision_config, SiglipVisionConfig):
            self.vision_config = vision_config
        elif isinstance(vision_config, dict):
            vision_config["model_type"] = (
                vision_config["model_type"] if "model_type" in vision_config else "siglip_vision_model"
            )
            self.vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        else:
            assert vision_pretrained is not None, f"You must provide text_config or vision_pretrained parameter"
            self.vision_config = CONFIG_MAPPING["siglip_vision_model"].from_pretrained(vision_pretrained)

        # Example of a parameter for projecting text/image features into a shared space
        self.projection_dim = projection_dim

        # For clarity, you may set a default architecture name or adapt as you see fit
        if not hasattr(self, "architectures"):
            self.architectures = ["BgeGemma2MultimodalModel"]
        super().__init__(**kwargs)

# # Optionally register the config in CONFIG_MAPPING if you wish to
# # use an AutoConfig-like pattern. For example:
# CONFIG_MAPPING.register("bge-gemma2-multimodal", BgeGemma2MultimodalConfig)
