import unittest

import torch
from transformers import AutoTokenizer

from config import settings
from verb_cluster.lightning.models import MaskedModel


class MaskedModelTest(unittest.TestCase):
    def setUp(self):
        pretrained_model_name_or_path = settings["pretrained_model_name_or_path"]
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = MaskedModel(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )

        sentence = "Gene induce disease"
        masked_sent = "Gene [MASK] disease"

        encoded_sentence = tokenizer(sentence, masked_sent, return_tensors="pt")

        masked_index = torch.zeros_like(encoded_sentence["input_ids"])
        verb_ids = torch.zeros((1))
        masked_index[0, 6] = 1
        verb_ids[0] = 1
        verb_ids = verb_ids.long()

        self.input_ids = encoded_sentence["input_ids"]
        self.token_type_ids = encoded_sentence["token_type_ids"]
        self.attention_mask = encoded_sentence["attention_mask"]
        self.masked_index = masked_index
        self.verb_ids = verb_ids

    def test_generate_transformer_representations(self):
        outputs, output_prediction = self.model._generate_transformer_representations(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            token_type_ids=self.token_type_ids,
        )

        assert outputs.shape[0] == 1
        assert outputs.shape[1] == 9

        assert output_prediction.shape[0] == 1
        assert output_prediction.shape[1] == 9

    def test_generate_sentence_representations(self):
        transformer_outputs = torch.randn(1, 9, 768)

        sentence_representation = self.model._generate_sentence_representations(
            attention_mask=self.attention_mask,
            token_type_ids=self.token_type_ids,
            token_index=1,
            transformer_outputs=transformer_outputs,
        )
        assert sentence_representation.shape[0] == 1
        assert sentence_representation.shape[1] == 768

    def test_mask_token_prediction(self):
        output_prediction = torch.arange(0, 9)
        output_prediction = output_prediction.unsqueeze(0).unsqueeze(2)
        output_prediction = output_prediction.repeat(1, 1, 100)

        verb_prediction = self.model._get_masked_token_prediction(
            mask_index=self.masked_index, output_prediction=output_prediction
        )

        assert verb_prediction[0, 0] == 6

    def test_model(self):
        loss, output, output_softmax = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            token_type_ids=self.token_type_ids,
            mask_index=self.masked_index,
            first_verb_token=self.verb_ids,
        )

        assert len(loss.shape) == 0
        assert output.shape[0] == 1
        assert output_softmax.shape[0] == 1
