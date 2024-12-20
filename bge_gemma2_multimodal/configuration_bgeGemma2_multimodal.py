"""BGE Multimodal configuration"""
import warnings

from transformers import PretrainedConfig, CONFIG_MAPPING
from transformers.models.gemma2 import Gemma2Config
from transformers.models.siglip import SiglipVisionConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


# Adapt from: transformers.models.paligemma.configuration_paligemma.PaliGemmaConfig
class BgeGemma2MultimodalConfig(PretrainedConfig):

    model_type = "bgemultimodal"
    sub_configs = {"text_config": Gemma2Config, "vision_config": SiglipVisionConfig}

    def __init__(
            self,
            vision_config: SiglipVisionConfig = None,
            text_config: Gemma2Config = None,
            ignore_index=-100,
            image_token_index=256000,
            vocab_size=257152,
            projection_dim=2048,
            hidden_size=2048,
            **kwargs,
            ):
        self._ignore_index = ignore_index
        self.image_token_index = image_token_index
        self._vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False

        if isinstance(self.vision_config, dict):
            vision_config["model_type"] = (
                vision_config["model_type"] if "model_type" in vision_config else "siglip_vision_model"
            )
            self.vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            self.vision_config = CONFIG_MAPPING["siglip_vision_model"](
                    intermediate_size=4096,
                    hidden_size=1152,
                    patch_size=14,
                    image_size=224,
                    num_hidden_layers=27,
                    num_attention_heads=16,
                    vocab_size=257152,
                    vision_use_head=False,
                    )

        self.text_config = text_config
        if isinstance(self.text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "gemma"
            self.text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            self.text_config = Gemma2Config.from_pretrained("BAAI/bge-multilingual-gemma2")
        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim
        super().__init__(**kwargs)

    @property
    def ignore_index(self):
        warnings.warn(
                "The `igno re_index` attribute is deprecated and will be removed in v4.47.",
                FutureWarning,
                )
        return self._ignore_index

    @ignore_index.setter
    def ignore_index(self, value):
        self._ignore_index = value

    def to_dict(self):
        output = super().to_dict()
        output.pop("_ignore_index", None)
        return output
