"""BGE Multimodal configuration"""
from typing import Union

from transformers import PretrainedConfig, CONFIG_MAPPING
from transformers.models.gemma2 import Gemma2Config
from transformers.models.siglip import SiglipVisionConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class BgeGemma2MultimodalConfig(PretrainedConfig):

    model_type = "bgemultimodal"
    sub_configs = {"text_config": Gemma2Config, "vision_config": SiglipVisionConfig}

    def __init__(
            self,
            text_config: Union[Gemma2Config, dict] = None,
            vision_config: Union[SiglipVisionConfig, dict] = None,
            text_pretrained: str = "BAAI/bge-multilingual-gemma2",
            vision_pretrained: str = "google/siglip-base-patch16-224",
            projection_dim=2048,
            image_token_index=256000,
            **kwargs
            ):
        """
        Initializes a multimodal model configuration combining text and vision modules.

        This constructor sets up the configuration for the text and vision modules, along with
        the projection dimension for feature representations. It allows configuration
        objects or pre-trained model names for both text and vision modules.

        Args:
            text_config (Union[Gemma2Config, dict]): The configuration for the text module.
                Can be an instance of `Gemma2Config` or a dictionary.
            vision_config (Union[SiglipVisionConfig, dict]): The configuration for the vision module.
                Can be an instance of `SiglipVisionConfig` or a dictionary.
            text_pretrained (str): The name of the pre-trained text model to use. Defaults
                to "BAAI/bge-multilingual-gemma2".
            vision_pretrained (str): The name of the pre-trained vision model to use. Defaults
                to "google/siglip-base-patch16-224".
            projection_dim (int): The dimensionality of the projection layer for aligning
                text and vision features in the same space. Defaults to 2048.
            **kwargs: Additional keyword arguments for further customization.

        Raises:
            AssertionError: If neither `text_config` nor `text_pretrained` is provided.
            AssertionError: If neither `vision_config` nor `vision_pretrained` is provided.
        """

        super().__init__(**kwargs)
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
            self.vision_config["model_type"] = (
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

    @classmethod
    def from_text_vision_configs(
            cls,
            text_config: Gemma2Config,
            vision_config: SiglipVisionConfig,
            **kwargs
            ):
        """
        Instantiate a `BgeGemma2MultimodalConfig` from Gemma2 + SigLIP Vision configs.

        Args:
            text_config (`Gemma2Config`): The config for the text encoder.
            vision_config (`SiglipVisionConfig`): The config for the vision encoder.
            kwargs: Additional keyword arguments for the multimodal config.

        Returns:
            `BgeGemma2MultimodalConfig`: A new multimodal configuration object.
        """
        return cls(
                text_config=text_config.to_dict(),
                vision_config=vision_config.to_dict(),
                **kwargs
                )

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Overridden from `PretrainedConfig`.
        Includes the text and vision sub-config dictionaries.
        """
        output = super().to_dict()
        # Convert sub-configs to dicts for serialization
        output["text_config"] = self.text_config.to_dict()
        output["vision_config"] = self.vision_config.to_dict()
        output["projection_dim"] = self.projection_dim
        return output


# Optionally register the config in CONFIG_MAPPING if you wish to
# use an AutoConfig-like pattern. For example:
CONFIG_MAPPING.register("bge-gemma2-multimodal", BgeGemma2MultimodalConfig)
