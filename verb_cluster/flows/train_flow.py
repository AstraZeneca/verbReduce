import os

import pandas as pd
from loguru import logger
from prefect import flow, task
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from config import settings
from verb_cluster.lightning import MaskedDataModule, MaskedPlModel
from verb_cluster.tasks import (prepare_masked_data,
                                prepare_masked_tokenized_data)


def train_fn():
    """Compose the training flow"""

    logger.info(f"Dynaconf environment variable: {os.getenv('ENV_FOR_DYNACONF')}")
    logger.info(f"Training model from data: {settings['path']['data_path']}")

    entailment_callback = ModelCheckpoint(
        monitor="entailment_percentage",
        dirpath=settings["path"]["checkpoint_path"],
        filename="{task_name}-{epoch:02d}-{entailment_percentage:.2f}-{sub_verb_count:0.2f}",
        save_top_k=3,
        mode="max",
    )
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=settings["path"]["logging_dir"])
    trainer = Trainer(
        accelerator=settings["accelerator"],
        max_epochs=settings["hyperparameters"]["max_epochs"],
        val_check_interval=settings["config_param"]["val_check_interval"],
        callbacks=[entailment_callback],
        logger=tb_logger,
        devices=1
        if settings["accelerator"] == "cpu" or settings["accelerator"] == "mps"
        else -1,
    )

    @task
    def init_data_module(masked_data: pd.DataFrame) -> MaskedDataModule:
        """Initializes the datamodule for prediction

        Args:
            masked_data (pd.DataFrame): Dataframe with masked sentences

        Returns:
            MaskedDataModule: Returns the initialized datamodule
        """

        masked_data_module = MaskedDataModule(
            df=masked_data,
            test_split=settings["config_param"]["test_split"],
            seed=settings["hyperparameters"]["seed"],
            num_workers=settings["config_param"]["num_workers"],
        )
        return masked_data_module

    @task
    def run_train(masked_data_module: MaskedDataModule) -> None:
        """Run training on the data in the datamodule

        Args:
            masked_data_module (MaskedDataModule): Datamodule with train/test data
        """
        model = MaskedPlModel(
            pretrained_model_name_or_path=settings["pretrained_model_name_or_path"],
            british_spellings_json_path=settings["british_spellings_json_path"],
            test_df=masked_data_module.test_df,
            **settings["hyperparameters"],
        )
        trainer.fit(model, masked_data_module)

    @flow
    def train_flow():
        """Training flow"""

        df = prepare_masked_data(
            data_path=settings["path"]["data_path"],
            pretrained_model_name_or_path=settings["pretrained_model_name_or_path"],
            verb_frequency_threshold=settings["hyperparameters"][
                "verb_frequency_threshold"
            ],
        )
        masked_df = prepare_masked_tokenized_data(
            df=df,
            pretrained_model_name_or_path=settings["pretrained_model_name_or_path"],
            max_seq_length=settings["hyperparameters"]["max_seq_length"],
        )
        masked_data_module = init_data_module(masked_df)
        run_train(masked_data_module)

    train_flow()


if __name__ == "__main__":
    train_fn()
