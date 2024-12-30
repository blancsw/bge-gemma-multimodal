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
    if os.path.exists("bge_gemma2_multimodal_hub_files"):
        do_init = click.confirm(
                "The folder 'bge_gemma2_multimodal_hub_files' exist. Do you want to overwrite ?")
    else:
        do_init = True
    if do_init:
        # Use Click's confirm for Yes/No input
        if click.confirm("Export the tokenizer ?"):
            tokenizer = GemmaTokenizerFast.from_pretrained(BASED_EMBEDDING_MODEL)
            tokenizer.save_pretrained("./bge_gemma2_multimodal_hub_files")
            print("""
            Edit added_tokens.json, special_tokens_map.json, tokenizer.json, tokenizer_config.json 
            to add <image> token use one of the <unusedX> token""")
            input("Press enter when finished editing.")

        if click.confirm("Export models config ?"):
            text_config = Gemma2Config.from_pretrained(BASED_EMBEDDING_MODEL)
            vision_config = SiglipVisionConfig.from_pretrained(BASED_VISION_MODEL)
            vision_config.projection_dim = vision_config.hidden_size * 2
            vision_config.num_positions = (vision_config.image_size // vision_config.patch_size) ** 2
            vision_config.num_image_tokens = vision_config.num_positions
            config = BgeGemma2MultimodalConfig(vision_config=vision_config,
                                               vision_pretrained=BASED_VISION_MODEL,
                                               projection_dim=vision_config.projection_dim,
                                               image_token_index=255999,
                                               text_config=text_config,
                                               text_pretrained=BASED_EMBEDDING_MODEL)
            config.save_pretrained("./bge_gemma2_multimodal_hub_files")

        if click.confirm("Export the processor ?"):
            config = BgeGemma2MultimodalConfig.from_pretrained("./bge_gemma2_multimodal_hub_files")
            image_processor = SiglipImageProcessor.from_pretrained(BASED_VISION_MODEL)
            tokenizer = GemmaTokenizerFast.from_pretrained("./bge_gemma2_multimodal_hub_files")

            processor = BgeGemma2MultimodalProcessor(image_processor=image_processor,
                                                     image_seq_length=config.vision_config.num_image_tokens,
                                                     tokenizer=tokenizer)
            processor.save_pretrained("./bge_gemma2_multimodal_hub_files")

    # else:
    #     tokenizer = GemmaTokenizerFast.from_pretrained("bge_gemma2_multimodal_hub_files")
    #     assert 255999 == tokenizer.convert_tokens_to_ids("<vision>")
    #     processor = BgeGemma2MultimodalProcessor.from_pretrained("bge_gemma2_multimodal_hub_files")
    #     click.echo("Processor initialized successfully.")


if __name__ == "__main__":
    main()
