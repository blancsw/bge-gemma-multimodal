import os

from transformers import GemmaTokenizerFast, SiglipImageProcessor

from bge_gemma2_multimodal.processing_bgeGemma2_multimodal import BgeGemma2MultimodalProcessor

if not os.path.exists("bge_gemma2_multimodal_hub_files"):
    tokenizer = GemmaTokenizerFast.from_pretrained("BAAI/bge-multilingual-gemma2")
    tokenizer.save_pretrained("./bge_gemma2_multimodal_hub_files")
    print("Edit added_tokens.json, special_tokens_map.json, tokenizer.json, tokenizer_config.json")
    input("Press enter when finish")
    image_processor = SiglipImageProcessor.from_pretrained("google/siglip-base-patch16-224")
    tokenizer = GemmaTokenizerFast.from_pretrained("./bge_gemma2_multimodal_hub_files")

    processor = BgeGemma2MultimodalProcessor(image_processor=image_processor, tokenizer=tokenizer)
    processor.save_pretrained("./bge_gemma2_multimodal_hub_files")
else:
    tokenizer = GemmaTokenizerFast.from_pretrained("bge_gemma2_multimodal_hub_files")
    assert 255999 == tokenizer.convert_tokens_to_ids("<vision>")
    processor = BgeGemma2MultimodalProcessor.from_pretrained("bge_gemma2_multimodal_hub_files")
    print()
