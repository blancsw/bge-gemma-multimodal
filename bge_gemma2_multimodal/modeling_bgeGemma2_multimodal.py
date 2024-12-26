from dataclasses import dataclass
from typing import Optional, Union, Tuple, List

import torch
from torch import nn
from transformers import PreTrainedModel, SiglipVisionModel, Gemma2Model, HybridCache, Cache
from transformers.utils import logging, ModelOutput

from .configuration_bgeGemma2_multimodal import BgeGemma2MultimodalConfig

logger = logging.get_logger(__name__)


@dataclass
class BgeGemma2MultimodalModelOutputWithPast(ModelOutput):
    """Output structure for model predictions and intermediate states.

        This class holds and organizes the outputs of a multimodal model.
        It is used to capture both final outputs such as logits and loss,
        as well as intermediate states like hidden states and attentions,
        which can be useful for analysis or further processing. The
        "past_key_values" attribute can store cached information to optimize
        decoder-based applications.

        Attributes:
            loss (Optional[torch.FloatTensor]): Training loss, if applicable. This is
                typically used during model optimization.
            logits (torch.FloatTensor): Final prediction scores representing
                unnormalized probabilities for each class or token.
            past_key_values (Optional[Union[List[torch.FloatTensor], Cache]]): Cached
                past key-value pairs used in decoder attention mechanisms to improve
                efficiency during generation tasks.
            hidden_states (Optional[Tuple[torch.FloatTensor]]): Hidden states from
                each layer of the model. These states can be used for insights or
                fine-grained processing.
            attentions (Optional[Tuple[torch.FloatTensor]]): Attention weights from
                the model at each layer. These can provide interpretability for
                which input tokens or features were prominent in predictions.
            image_hidden_states (Optional[torch.FloatTensor]): Hidden states derived
                from image-based inputs. These are specific to the multimodal
                processing stage of the model.
            last_hidden_state (torch.FloatTensor): The final hidden state produced
                by the model, which serves as the primary output representation for
                downstream tasks.
    ```
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None


class BgeGemma2MultiModalProjector(nn.Module):
    def __init__(self, config: BgeGemma2MultimodalConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear(image_features)

        return hidden_states


class BgeGemma2MultimodalPreTrainedModel(PreTrainedModel):
    config_class = BgeGemma2MultimodalConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BgeGemma2MultimodalProjector"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = False
    _supports_static_cache = True

    def _init_weights(self, module):
        # important: this ported version of PaliGemmaisn't meant for training from scratch - only
        # inference and fine-tuning
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @classmethod
    def _check_and_enable_sdpa(cls, config, hard_check_only: bool = False):
        """
        Overloads `PreTrainedModel._check_and_enable_sdpa` so as to DISABLE torch SDPA by default on Gemma2 models.
        SDPA reduces the model performance on Gemma2 because of the logits softcapping.
        """
        config = super()._check_and_enable_sdpa(config, hard_check_only=hard_check_only)

        # if using the default path -> swap sdpa by eager
        if not hard_check_only and config._attn_implementation == "sdpa":
            config._attn_implementation = "eager"

        return config


class BgeGemma2MultimodalModel(BgeGemma2MultimodalPreTrainedModel):
    def __init__(self, config: BgeGemma2MultimodalConfig):
        super().__init__(config)
        self.config = config

        # 1. Create the text model from Gemma2 config
        self.text_model = Gemma2Model._from_config(self.config.text_config)

        # 2. Create the vision model from Siglip config
        self.vision_model = SiglipVisionModel._from_config(self.config.vision_config)

        self.multi_modal_projector = BgeGemma2MultiModalProjector(config)

        # If you would like to tie weights or do other advanced logic, do it here
        self.post_init()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.get_input_embeddings with Llava->PaliGemma
    def get_input_embeddings(self):
        return self.text_model.get_input_embeddings()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.set_input_embeddings with Llava->PaliGemma
    def set_input_embeddings(self, value):
        self.text_model.set_input_embeddings(value)

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.get_output_embeddings with Llava->PaliGemma
    def get_output_embeddings(self):
        return self.text_model.get_output_embeddings()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.set_output_embeddings with Llava->PaliGemma
    def set_output_embeddings(self, new_embeddings):
        self.text_model.set_output_embeddings(new_embeddings)

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.set_decoder with Llava->PaliGemma
    def set_decoder(self, decoder):
        self.text_model.set_decoder(decoder)

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.get_decoder with Llava->PaliGemma
    def get_decoder(self):
        return self.text_model.get_decoder()

    def get_image_features(self, pixel_values: torch.FloatTensor):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
               The tensors corresponding to the input images.
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        image_outputs = self.vision_model(pixel_values)
        selected_image_feature = image_outputs.last_hidden_state
        image_features = self.multi_modal_projector(selected_image_feature)
        image_features = image_features / (self.config.text_config.hidden_size ** 0.5)
        return image_features

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            pixel_values: torch.FloatTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[HybridCache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            ) -> Union[Tuple, BgeGemma2MultimodalModelOutputWithPast]:

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds\n"
                             "For vision only embedding add <vision> token into your prompt")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both pixel_values and inputs_embeds at the same time, "
                             "and must specify either one")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # TODO
        # is_training = token_type_ids is not None and labels is not None

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                    )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0) + 1  # Paligemma positions are 1-indexed

        # Merge text and images
        image_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)

            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
            if inputs_embeds[special_image_mask].numel() != image_features.numel():
                image_tokens_in_text = torch.sum(input_ids == self.config.image_token_index)
                raise ValueError(
                        f"Number of images does not match number of special image tokens in the input text. "
                        f"Got {image_tokens_in_text} image tokens in the text but {image_features.shape[0] * image_features.shape[1]} "
                        "tokens from image embeddings."
                        )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # mask out pad-token-ids in labels for BC
        if labels is not None and self.pad_token_id in labels:
            logger.warning_once(
                    "`labels` contains `pad_token_id` which will be masked with `config.ignore_index`. ",
                    "You have to mask out `pad_token_id` when preparing `labels`, this behavior will be removed in v.4.46.",
                    )
            labels = torch.where(input_ids == self.pad_token_id, self.config.ignore_index, labels)

        # TODO see if need to keep this
        # causal_mask = self._update_causal_mask(
        #         attention_mask, token_type_ids, past_key_values, cache_position, input_ids, inputs_embeds, is_training
        #         )
        causal_mask = None
        outputs = self.language_model(
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position)

        logits = outputs.logits
        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1]:].to(logits.device)
                shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()

            flat_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(flat_logits, flat_labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return BgeGemma2MultimodalModelOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                image_hidden_states=image_features,
                )
