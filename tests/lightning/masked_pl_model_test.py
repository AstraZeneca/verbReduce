import os
import unittest

from pytorch_lightning import Trainer

from config import settings
from verb_cluster.lightning import MaskedDataModule, MaskedPlModel
from verb_cluster.tasks import (prepare_masked_data,
                                prepare_masked_tokenized_data)

os.environ["ENV_FOR_DYNACONF"] = "test"


class MasedPLModelTest(unittest.TestCase):
    def test_maksed_pl_module(self):
        data_path = settings["path"]["data_path"]
        pretrained_model_name_or_path = settings["pretrained_model_name_or_path"]
        verb_frequency_threshold = settings["hyperparameters"][
            "verb_frequency_threshold"
        ]
        british_spellings_json_path = settings["british_spellings_json_path"]

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

        pl_model = MaskedPlModel(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            british_spellings_json_path=british_spellings_json_path,
            test_df=df,
            verb_frequency_threshold=1,
        )

        trainer = Trainer(max_epochs=1)
        predictions = trainer.predict(pl_model, datamodule)

        assert (len(predictions[0]["predictions"])) == 26

        trainer.fit(pl_model, datamodule)
