"""
Processor class for BgeGemma2Multimodal.
"""

from typing import List, Union, Optional

from transformers import GemmaTokenizerFast, SiglipImageProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, is_valid_image
from transformers.processing_utils import (
    ProcessorMixin,
    Unpack,
    _validate_images_text_input_order,
    ProcessingKwargs,
    TextKwargs,
    ImagesKwargs,
    )
from transformers.tokenization_utils_base import (
    PreTokenizedInput,
    TextInput,
    )
from transformers.utils import logging

logger = logging.get_logger(__name__)


class BgeGemma2MultimodalTextKwargs(TextKwargs):
    instruct: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]]
    return_formated_prompt: Optional[bool]


class BgeGemma2MultimodalImagesKwargs(ImagesKwargs):
    do_convert_rgb: Optional[bool]


class BgeGemma2MultimodalProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: BgeGemma2MultimodalTextKwargs
    images_kwargs: BgeGemma2MultimodalImagesKwargs
    return_formated_prompt: Optional[bool]
    _defaults = {
        "text_kwargs":   {
            "padding":                False,
            "return_formated_prompt": False,
            },
        "images_kwargs": {
            "data_format": "channels_first",
            },

        }


def remove_tokens(text: str, tokens: list) -> str:
    """
    Removes all occurrences of specified tokens from the given text.

    Args:
        text (str): The input text to process.
        tokens (list): List of tokens (substrings) to remove from the text.

    Returns:
        str: The cleaned text with all tokens removed.
    """
    for token in tokens:
        text = text.replace(token, '')
    return text


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
    IMAGE_TOKEN = "<vision>"
    QUERY_TOKEN = "<query>"
    TEXT_INSTRUCT_TOKEN = "<instruct>"
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
            self.image_seq_length = 256
        else:
            if not hasattr(image_processor, "image_seq_length"):
                raise ValueError("Image processor is missing an `image_seq_length` attribute.")
            # Added attribut to the SiglipImageProcessor
            self.image_seq_length = image_processor.image_seq_length
        if tokenizer is None:
            tokenizer = GemmaTokenizerFast.from_pretrained("./bge_gemma2_multimodal_hub_files")

        assert self.IMAGE_TOKEN in tokenizer.all_special_tokens
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        super().__init__(image_processor, tokenizer)

    def format_text_input(
            self,
            text: TextInput = None,
            image=None,
            instruct: TextInput = None
            ) -> TextInput:
        assert text or image, "You must specify either `text` or/and `image`."
        special_tokens = [self.IMAGE_TOKEN, self.TEXT_INSTRUCT_TOKEN, self.QUERY_TOKEN]
        # Image only embedding
        if text is None:
            # Query image embedding
            if isinstance(instruct, str):
                instruct = remove_tokens(instruct, special_tokens)
                # format: <vision><instruct>oposite images
                text = f"{self.IMAGE_TOKEN * self.image_seq_length}{self.TEXT_INSTRUCT_TOKEN}{instruct}"

            # index image embedding
            else:
                text = self.IMAGE_TOKEN
        # text only embedding
        elif image is None:
            text = remove_tokens(text, special_tokens)
            # Query text embedding
            if isinstance(instruct, str):
                # remove all special tokens to simplify proicessing
                instruct = remove_tokens(instruct, special_tokens)
                # format: <instruct>Given a web search query, retrieve...<query>My last bill
                text = f"{self.TEXT_INSTRUCT_TOKEN}{instruct}{self.QUERY_TOKEN}{text}"
        # Image and text embedding
        else:
            text = remove_tokens(text, special_tokens)
            # Query text/image embedding
            if isinstance(instruct, str):
                # remove all special tokens to simplify proicessing
                instruct = remove_tokens(instruct, special_tokens)
                # format: <vision><instruct>Given a web search query, retrieve...<query>Find similar color image
                text = f"{self.IMAGE_TOKEN}{self.TEXT_INSTRUCT_TOKEN}{instruct}{self.QUERY_TOKEN}{text}"
            # text and image embedding
            else:
                # format: <vision>Find similar color image
                text = f"{self.IMAGE_TOKEN}{text}"
        return text

    def __call__(
            self,
            images: ImageInput = None,
            text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
            audio=None,
            videos=None,
            **kwargs: Unpack[BgeGemma2MultimodalProcessorKwargs],
            ) -> BatchFeature:
        # check if images and text inputs are reversed for BC
        images, text = _validate_images_text_input_order(images, text)

        output_kwargs = self._merge_kwargs(
                BgeGemma2MultimodalProcessorKwargs,
                tokenizer_init_kwargs=self.tokenizer.init_kwargs,
                **kwargs,
                )
        instruct = output_kwargs["text_kwargs"].pop("instruct", None)
        return_formated_prompt = output_kwargs["text_kwargs"].pop("return_formated_prompt", False)

        if not (images or text):
            raise ValueError("You have to specify either `text` or `images`.")

        if text is None:
            text = [self.format_text_input(None, images, instruct)]
        # Ensure `text` is a list of inputs
        elif _is_str_or_image(text):
            text = [self.format_text_input(text, images, instruct)]
        else:
            text = [self.format_text_input(t, images, instruct) for t in text]

        # Check if text and images are both provided and have mismatched lengths
        if text and images and isinstance(text, List) and isinstance(images, List) and len(images) != len(text):
            raise ValueError(
                    f"Received {len(images)} images for {len(text)} prompts. Each prompt should be associated with an image or list of images."
                    )

        # preprocess images
        if images:
            # make a nested list of lists to be able to iterate over the images and text below
            if is_valid_image(images):
                images = [[images]]
            elif isinstance(images, list) and is_valid_image(images[0]):
                images = [[image] for image in images]
            elif not (isinstance(images, list) and isinstance(images[0], list) and is_valid_image(images[0][0])):
                raise ValueError("images must be an image, list of images or list of list of images")
            # TODO je pensse pas nessesaire
            images = make_batched_images(images)
            pixel_values = self.image_processor(images, **output_kwargs["images_kwargs"])["pixel_values"]
        else:
            pixel_values = None

        # max_length has to account for the image tokens
        if output_kwargs["text_kwargs"].get("max_length", None) is not None:
            output_kwargs["text_kwargs"]["max_length"] += self.image_seq_length

        inputs = self.tokenizer(
                text,
                **output_kwargs["text_kwargs"],
                )
        data = {**inputs, "pixel_values": pixel_values}
        if return_formated_prompt:
            data["formated_prompt"] = text
        return BatchFeature(data=data)

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
