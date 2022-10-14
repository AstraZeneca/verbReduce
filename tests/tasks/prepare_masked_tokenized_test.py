import os
import unittest

from config import settings
from verb_cluster.tasks import (prepare_masked_data,
                                prepare_masked_tokenized_data)

os.environ["ENV_FOR_DYNACONF"] = "test"


class PrepareMaskedTokenizedDataTest(unittest.TestCase):
    def test_masked_data_with_words_not_in_vocab(self):
        data_path = settings["path"]["data_path"]
        pretrained_model_name_or_path = settings["pretrained_model_name_or_path"]
        verb_frequency_threshold = settings["hyperparameters"][
            "verb_frequency_threshold"
        ]

        df = prepare_masked_data.fn(
            data_path=data_path,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            verb_frequency_threshold=verb_frequency_threshold,
        )
        assert df.shape[0] == 28
        df = prepare_masked_tokenized_data.fn(
            df=df, pretrained_model_name_or_path=pretrained_model_name_or_path
        )
        assert df.shape[0] == 26

    def test_masked_data_with_words_in_vocab(self):
        data_path = settings["path"]["data_path"]
        pretrained_model_name_or_path = settings["pretrained_model_name_or_path"]
        verb_frequency_threshold = settings["hyperparameters"][
            "verb_frequency_threshold"
        ]

        df = prepare_masked_data.fn(
            data_path=data_path,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            verb_frequency_threshold=verb_frequency_threshold,
        )
        df = df[df["verb_form"] != "co-expressed"]
        assert df.shape[0] == 26
        df = prepare_masked_tokenized_data.fn(
            df=df, pretrained_model_name_or_path=pretrained_model_name_or_path
        )
        assert df.shape[0] == 26

    def test_masked_data_with_index_error(self):
        data_path = settings["path"]["data_path"]
        pretrained_model_name_or_path = settings["pretrained_model_name_or_path"]
        verb_frequency_threshold = settings["hyperparameters"][
            "verb_frequency_threshold"
        ]

        df = prepare_masked_data.fn(
            data_path=data_path,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            verb_frequency_threshold=verb_frequency_threshold,
        )
        df.at[2, "masked_sent"] = "Sentence with no mask"
        df = prepare_masked_tokenized_data.fn(
            df=df, pretrained_model_name_or_path=pretrained_model_name_or_path
        )
        assert df.shape[0] == 25
