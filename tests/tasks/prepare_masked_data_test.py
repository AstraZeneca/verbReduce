import os
import unittest

from config import settings
from verb_cluster.tasks import prepare_masked_data

os.environ["ENV_FOR_DYNACONF"] = "test"


class PrepareMasekdDataTest(unittest.TestCase):
    def test_masked_data(self):
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
        assert "verb_form" in df.columns
        assert "masked_sent" in df.columns
        data = df.iloc[0]
        assert data["sentence"].find("targets") != -1
        assert data["masked_sent"].find("targets") == -1
        assert data["masked_sent"].find("[MASK]") != -1

    def test_masked_data_with_sample(self):
        data_path = settings["path"]["data_path"]
        pretrained_model_name_or_path = settings["pretrained_model_name_or_path"]
        verb_frequency_threshold = settings["hyperparameters"][
            "verb_frequency_threshold"
        ]

        df = prepare_masked_data.fn(
            data_path=data_path,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            verb_frequency_threshold=verb_frequency_threshold,
            sample_size=0.6,
        )
        assert df.shape[0] == 16
