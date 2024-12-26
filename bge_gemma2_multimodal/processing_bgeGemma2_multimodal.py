"""
Processor class for BgeGemma2Multimodal.
"""

from typing import List, Union

from transformers import GemmaTokenizerFast, SiglipImageProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, is_valid_image
from transformers.models.paligemma.processing_paligemma import PaliGemmaProcessorKwargs
from transformers.processing_utils import (
    ProcessorMixin,
    Unpack,
    _validate_images_text_input_order,
    )
from transformers.tokenization_utils_base import (
    PreTokenizedInput,
    TextInput,
    )
from transformers.utils import logging

logger = logging.get_logger(__name__)

IMAGE_TOKEN = "<vision>"


# Copied from transformers.models.idefics2.processing_idefics2.is_url
def is_url(val) -> bool:
    return isinstance(val, str) and val.startswith("http")


# Copied from transformers.models.idefics2.processing_idefics2.is_image_or_image_url
def is_image_or_image_url(elem):
    return is_url(elem) or is_valid_image(elem)


def _is_str_or_image(elem):
    return isinstance(elem, (str)) or is_image_or_image_url(elem)


def build_string_from_input(prompt, bos_token, image_seq_len, image_token, num_images):
    """
    Builds a string from the input prompt and image tokens.
    For example, for the call:
    build_string_from_input(
        prompt="Prefix str"
        bos_token="<s>",
        image_seq_len=3,
        image_token="<im>",
    )
    The output will be:
    "<im><im><im><s>Initial str"
    Args:
        prompt (`List[Union[str, ImageInput]]`): The input prompt.
        bos_token (`str`): The beginning of sentence token.
        image_seq_len (`int`): The length of the image sequence.
        image_token (`str`): The image token.
        num_images (`int`): Number of images in the prompt.
    """
    return f"{image_token * image_seq_len * num_images}{bos_token}{prompt}\n"


# Copied from transformers.models.llava_next.image_processing_llava_next.make_batched_images
def make_batched_images(images) -> List[List[ImageInput]]:
    """
    Accepts images in list or nested list format, and makes a list of images for preprocessing.

    Args:
        images (`Union[List[List[ImageInput]], List[ImageInput], ImageInput]`):
            The input image.

    Returns:
        list: A list of images.
    """
    if isinstance(images, (list, tuple)) and isinstance(images[0], (list, tuple)) and is_valid_image(images[0][0]):
        return [img for img_list in images for img in img_list]

    elif isinstance(images, (list, tuple)) and is_valid_image(images[0]):
        return images

    elif is_valid_image(images):
        return [images]

    raise ValueError(f"Could not make batched video from {images}")


# Adapt from transformers.models.paligemma.processing_paligemma.PaliGemmaProcessor
class BgeGemma2MultimodalProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "SiglipImageProcessor"
    tokenizer_class = ("GemmaTokenizer", "GemmaTokenizerFast")

    def __init__(
            self,
            image_processor: SiglipImageProcessor = None,
            tokenizer: GemmaTokenizerFast = None,
            **kwargs,
            ):
        if image_processor is None:
            image_processor = SiglipImageProcessor.from_pretrained("google/siglip-base-patch16-224")
        if tokenizer is None:
            tokenizer = GemmaTokenizerFast.from_pretrained("./bge_gemma2_multimodal_hub_files")
        self.image_seq_length = 1
        assert IMAGE_TOKEN in tokenizer.all_special_tokens
        self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        super().__init__(image_processor, tokenizer)

    def __call__(
            self,
            images: ImageInput = None,
            text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
            audio=None,
            videos=None,
            **kwargs: Unpack[PaliGemmaProcessorKwargs],
            ) -> BatchFeature:
        # check if images and text inputs are reversed for BC
        images, text = _validate_images_text_input_order(images, text)

        output_kwargs = self._merge_kwargs(
                PaliGemmaProcessorKwargs,
                tokenizer_init_kwargs=self.tokenizer.init_kwargs,
                **kwargs,
                )
        suffix = output_kwargs["text_kwargs"].pop("suffix", None)

        return_token_type_ids = True if suffix is not None else False

        if images is None:
            raise ValueError("`images` are expected as arguments to a `PaliGemmaProcessor` instance.")
        if text is None:
            logger.warning_once(
                    "You are using PaliGemma without a text prefix. It will perform as a picture-captioning model."
                    )
            text = ""

        if _is_str_or_image(text):
            text = [text]
        elif isinstance(text, list) and _is_str_or_image(text[0]):
            pass

        if text is not None and images is not None:
            if not any(IMAGE_TOKEN in sample for sample in text):
                logger.warning(
                        "You are passing both `text` and `images` to `PaliGemmaProcessor`. The processor expects special "
                        "image tokens in the text, as many tokens as there are images per each text. It is recommended to "
                        f"add `{IMAGE_TOKEN}` tokens in the very beginning of your text. For this call, we will infer how many images "
                        "each text has and add special tokens."
                        )

                if isinstance(text, List) and isinstance(images, List):
                    if len(images) != len(text):
                        raise ValueError(
                                f"Received {len(images)} images for {len(text)} prompts. Each prompt should be associated with an image or list of images."
                                )

                # make a nested list of lists to be able to iterate over the images and text below
                if is_valid_image(images):
                    images = [[images]]
                elif isinstance(images, list) and is_valid_image(images[0]):
                    images = [[image] for image in images]
                elif not (isinstance(images, list) and isinstance(images[0], list) and is_valid_image(images[0][0])):
                    raise ValueError("images must be an image, list of images or list of list of images")

                if suffix is not None and _is_str_or_image(suffix):
                    suffix = [suffix]
                if suffix is not None:
                    suffix = [sfx + self.tokenizer.eos_token for sfx in suffix]

                input_strings = [
                    build_string_from_input(
                            prompt=prompt,
                            bos_token=self.tokenizer.bos_token,
                            image_seq_len=self.image_seq_length,
                            image_token=IMAGE_TOKEN,
                            num_images=len(image_list) if isinstance(image_list, list) else 1,
                            )
                    for prompt, image_list in zip(text, images)
                    ]
                images = make_batched_images(images)
            else:
                expanded_samples = []
                for sample in text:
                    expanded_sample = sample.replace(IMAGE_TOKEN, IMAGE_TOKEN * self.image_seq_length)
                    bos_rfind_index = expanded_sample.rfind(IMAGE_TOKEN)
                    bos_index = bos_rfind_index + len(IMAGE_TOKEN) if bos_rfind_index != -1 else 0
                    expanded_sample = (
                            expanded_sample[:bos_index] + self.tokenizer.bos_token + expanded_sample[bos_index:]
                    )
                    expanded_samples.append(expanded_sample)
                input_strings = [f"{sample}\n" for sample in expanded_samples]
        pixel_values = self.image_processor(images, **output_kwargs["images_kwargs"])["pixel_values"]

        # max_length has to account for the image tokens
        if output_kwargs["text_kwargs"].get("max_length", None) is not None:
            output_kwargs["text_kwargs"]["max_length"] += self.image_seq_length

        inputs = self.tokenizer(
                input_strings,
                text_pair=suffix,
                return_token_type_ids=return_token_type_ids,
                **output_kwargs["text_kwargs"],
                )

        return_data = {**inputs, "pixel_values": pixel_values}

        if return_token_type_ids:
            labels = inputs["input_ids"].masked_fill(inputs["token_type_ids"] == 0, -100)
            return_data.update({"labels": labels})
        return BatchFeature(data=return_data)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Gemma
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GemmaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Gemma
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GemmaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names with CLIP->PaliGemma
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
