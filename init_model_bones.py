"""
Script to ini the model config files
- tokenizer
- model config
- processor
"""
import os

import click
from transformers.models.gemma import GemmaTokenizerFast
from transformers.models.gemma2 import Gemma2Config
from transformers.models.siglip import SiglipImageProcessor, SiglipVisionConfig

from bge_gemma2_multimodal import BgeGemma2MultimodalConfig
from bge_gemma2_multimodal.processing_bgeGemma2_multimodal import BgeGemma2MultimodalProcessor


@click.command()
def main():
    BASED_VISION_MODEL = "google/siglip-base-patch16-224"
    BASED_EMBEDDING_MODEL = "BAAI/bge-multilingual-gemma2"
    SAVE_LOCAL_PATH = "./bge_gemma2_multimodal_hub_files"

    if os.path.exists(SAVE_LOCAL_PATH):
        do_init = click.confirm(
                "The folder 'bge_gemma2_multimodal_hub_files' exist. Do you want to overwrite ?")
    else:
        do_init = True
    if do_init:
        # Use Click's confirm for Yes/No input
        if click.confirm("Export the tokenizer ?"):
            # Export the base embedding model tokenizer
            tokenizer = GemmaTokenizerFast.from_pretrained(BASED_EMBEDDING_MODEL)
            tokenizer.save_pretrained(SAVE_LOCAL_PATH)
            print("""
            Edit added_tokens.json, special_tokens_map.json, tokenizer.json, tokenizer_config.json 
            to add <image> token use one of the <unusedX> token""")
            input("Press enter when finished editing.")

        if click.confirm("Export models config ?"):
            tokenizer = GemmaTokenizerFast.from_pretrained(SAVE_LOCAL_PATH)
            # Load the base embedding model config
            text_config = Gemma2Config.from_pretrained(BASED_EMBEDDING_MODEL)
            # Load the vision model config
            vision_config = SiglipVisionConfig.from_pretrained(BASED_VISION_MODEL)
            # projection dim need to match the embedding gemma2 features dim
            vision_config.projection_dim = text_config.hidden_size
            # Compute image features
            vision_config.num_positions = (vision_config.image_size // vision_config.patch_size) ** 2
            vision_config.num_image_tokens = vision_config.num_positions
            # Note used full to add extra multi head attention pooling:
            # transformers.models.siglip.modeling_siglip.SiglipMultiheadAttentionPoolingHead
            vision_config.vision_use_head = False
            # get id of the vision special token
            image_token_index = tokenizer.convert_tokens_to_ids(BgeGemma2MultimodalProcessor.IMAGE_TOKEN)
            config = BgeGemma2MultimodalConfig(vision_config=vision_config,
                                               vision_pretrained=BASED_VISION_MODEL,
                                               projection_dim=vision_config.projection_dim,
                                               image_token_index=image_token_index,
                                               text_config=text_config,
                                               text_pretrained=BASED_EMBEDDING_MODEL)
            config.save_pretrained(SAVE_LOCAL_PATH)

        if click.confirm("Export the processor ?"):
            # Load the config to get information
            config = BgeGemma2MultimodalConfig.from_pretrained(SAVE_LOCAL_PATH)
            # Get the base vision processor
            image_processor = SiglipImageProcessor.from_pretrained(BASED_VISION_MODEL)
            # Load the edit tokenizer
            tokenizer = GemmaTokenizerFast.from_pretrained(SAVE_LOCAL_PATH)
            # Build the main processor
            processor = BgeGemma2MultimodalProcessor(image_processor=image_processor,
                                                     image_seq_length=config.vision_config.num_image_tokens,
                                                     tokenizer=tokenizer)
            processor.save_pretrained(SAVE_LOCAL_PATH)


if __name__ == "__main__":
    main()
