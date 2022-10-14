import os
import unittest

from config import settings
from verb_cluster.lightning import MaskedDataModule
from verb_cluster.tasks import (prepare_masked_data,
                                prepare_masked_tokenized_data)

os.environ["ENV_FOR_DYNACONF"] = "test"


class MaskedDataModuleTest(unittest.TestCase):
    def test_maksed_datamodule(self):
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

        datamodule = MaskedDataModule(
            df=df, num_workers=settings["config_param"]["num_workers"]
        )
        datamodule.setup()

        assert len(datamodule.train) == 21
        assert len(datamodule.test) == 5
        assert len(datamodule.predict) == 26

        train_dataloader = datamodule.train_dataloader()

        for data in train_dataloader:
            assert len(data) == 11
