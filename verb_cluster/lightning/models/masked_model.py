from typing import Optional, Tuple

import torch
from torch import nn
from transformers import AutoModelForMaskedLM


def _mean_pooling(
    model_output: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Mean pooling of sentences

    Args:
        model_output (torch.Tensor): Model ouput from transformers
        attention_mask (torch.Tensor): Attention mask of sentence

    Returns:
        torch.Tensor: Sentence embedding tensor
    """
    token_embeddings = (
        model_output  # First element of model_output contains all token embeddings
    )
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class MaskedModel(nn.Module):
    """Self supervised masked language modeling augumented with mean pooling
    sentence cosine loss
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = "bert-large-uncased-whole-word-masking",
    ):
        """Init function

        Args:
            pretrained_model_name_or_path (str): Pretrained transformer model
                used to encode the sentence
        """
        super(MaskedModel, self).__init__()
        self.transformer_model = AutoModelForMaskedLM.from_pretrained(
            pretrained_model_name_or_path
        )
        self.softmax = nn.Softmax(dim=2)
        self.cross_loss = nn.CrossEntropyLoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()

    def _generate_transformer_representations(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """_generate_transformer_representations.

        Args:
            input_ids (Optional[torch.Tensor]): Input ids of encoded sentence
            attention_mask (Optional[torch.Tensor]): Attention mask of encoded sentence
            token_type_ids (Optional[torch.Tensor]): Token type Ids of encoded sentence

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Returns encoded final output embedding
                and token prediction
        """
        # Generate transformer representations
        outputs = self.transformer_model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )[0]
        output_prediction = self.transformer_model.cls(outputs)
        return outputs, output_prediction

    def _generate_sentence_representations(
        self,
        attention_mask: Optional[torch.Tensor],
        token_type_ids: Optional[torch.Tensor],
        transformer_outputs: torch.Tensor,
        token_index: int,
    ) -> torch.Tensor:
        combination_tokens = token_type_ids + attention_mask
        sent_mask = torch.where(
            combination_tokens == token_index,
            torch.ones_like(attention_mask),
            torch.zeros_like(attention_mask),
        )
        return _mean_pooling(transformer_outputs, sent_mask)

    def _get_masked_token_prediction(
        self, mask_index: torch.Tensor, output_prediction: torch.Tensor
    ) -> torch.Tensor:
        bool_mask_tensor = mask_index.bool()
        verb_prediction = output_prediction[bool_mask_tensor][
            torch.arange(output_prediction.shape[0]), :
        ]
        return verb_prediction

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        mask_index: Optional[torch.Tensor] = None,
        first_verb_token: torch.Tensor = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode the input ids to generate verb candidates

        Args:
            input_ids (Optional[torch.Tensor]): Input ids of encoded sentence
            attention_mask (Optional[torch.Tensor]): Attention mask of encoded sentence
            token_type_ids (Optional[torch.Tensor]): Token type Ids of encoded sentence
            mask_index (Optional[torch.Tensor]): Masked Tensor with 1 where the [MASK]
                token is found
            first_verb_token (torch.Tensor): First token of the verb (our approach at
                the moment only deal for verbs with one token)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: returns the loss, token
                prediction and token prediction softmax
        """

        outputs, output_prediction = self._generate_transformer_representations(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sent1_embedding = self._generate_sentence_representations(
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            transformer_outputs=outputs,
            token_index=1,
        )
        sent2_embedding = self._generate_sentence_representations(
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            transformer_outputs=outputs,
            token_index=2,
        )
        # Sentence mean pooling loss
        cos_labels = torch.ones((outputs.shape[0]))
        if outputs.get_device() >= 0:
            cos_labels = cos_labels.to(outputs.get_device())
        sentence_cosine_loss = self.cosine_loss(
            sent1_embedding, sent2_embedding, cos_labels
        )

        # Token prediction loss
        verb_prediction = self._get_masked_token_prediction(
            mask_index=mask_index, output_prediction=output_prediction
        )
        token_prediction_loss = self.cross_loss(verb_prediction, first_verb_token)

        # TODO: Currently both losses are equally weighted,
        # Need to try with different weights. Can be explore in future
        loss = sentence_cosine_loss + token_prediction_loss

        return (
            loss,
            outputs,
            output_prediction.softmax(dim=2),
        )
