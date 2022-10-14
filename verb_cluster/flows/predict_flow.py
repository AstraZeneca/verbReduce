import os
import pickle
from functools import reduce
from typing import Dict

import pandas as pd
from loguru import logger
from prefect import flow, task
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers

from config import settings
from verb_cluster.lightning import MaskedDataModule, MaskedPlModel
from verb_cluster.tasks import (check_entailment, prepare_masked_data,
                                prepare_masked_tokenized_data,
                                prepare_substitute_sentences, reduce_verbs)


def predict_fn():
    """Compose the prediction flow"""

    logger.info(f"Dynaconf environment variable: {os.getenv('ENV_FOR_DYNACONF')}")
    logger.info(
        "Predicting based on model"
        + f"from: {settings['prediction_model_path']}"
        + f" on data: {settings['path']['data_path']}"
    )
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=settings["path"]["logging_dir"])
    trainer = Trainer(
        accelerator=settings["accelerator"],
        max_epochs=settings["hyperparameters"]["max_epochs"],
        val_check_interval=settings["config_param"]["val_check_interval"],
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
    def run_predict(masked_data_module: MaskedDataModule) -> Dict:
        """Run the prediction on the datamoduels

        Args:
            masked_data_module (MaskedDataModule): Datamodule containing the
                prediction dataset

        Returns:
            Dict: Returns the predictions
        """
        model_path = settings["prediction_model_path"]

        model = MaskedPlModel(
            pretrained_model_name_or_path=settings["pretrained_model_name_or_path"],
            british_spellings_json_path=settings["british_spellings_json_path"],
            **settings["hyperparameters"],
        )
        if model_path is not None and not model_path == "None":
            logger.info(f"Loading model from {model_path}")
            updated_model = model.load_from_checkpoint(model_path)
        else:
            updated_model = model
            logger.warning(
                "Model path not provided."
                + f"Using pretrained_model: {settings['pretrained_model_name_or_path']}"
            )
        predictions = trainer.predict(updated_model, masked_data_module)
        predictions = reduce(lambda s, d: {**s, **d["predictions"]}, predictions, {})
        return predictions

    @task
    def save_predictions_to_file(
        df: pd.DataFrame,
        entailment_output: Dict,
        verb_mapping: Dict[str, str],
        predictions: Dict,
    ) -> None:
        """Save the lookup table and the datafram with substituted sentences
            in the prediction path. We also save the dataframe where
            the entailment does not hold in the sampled dataset.

        Args:
            df (pd.DataFrame): Dataframe with substituted sentences
            entailment_output (Dict): Output from entailment evaluation
            verb_mapping (Dict[str, str]): Verb mapping mapping source verb
                to target verb
            predictions (Dict): predictions generated from the self-supervised
                model
        """
        df = df.drop(
            ["verb_ids", "encoded_sentence", "verb_mask", "mask_index"], axis=1
        )

        missed_df_data = entailment_output["missed_df_data"]
        prediction_path = settings["path"]["prediction_path"]
        if not os.path.exists(prediction_path):
            os.makedirs(prediction_path)

        with open(f"{prediction_path}/predictions.pkl", "wb") as f:
            pickle.dump(predictions, f)

        df.to_csv(f"{prediction_path}/subs.csv")
        missed_df_data.to_csv(f"{prediction_path}/missed.csv")
        pd.DataFrame(
            verb_mapping.items(), columns=["source_verb", "target_verb"]
        ).to_csv(f"{prediction_path}/lut.csv")
        logger.success(f"Predictions saved to {settings['path']['prediction_path']}")

    @task
    def sample_eval(eval_df: pd.DataFrame) -> pd.DataFrame:
        """Sample test_count of data from original dataframe for evaluation

        Args:
            eval_df (pd.DataFrame): Dataframe to sample from

        Returns:
            pd.DataFrame: Returns the sample dataframe
        """

        return eval_df.groupby("original_verb", group_keys=False).apply(
            lambda x: x.sample(
                frac=min(eval_df.shape[0], settings["hyperparameters"]["test_count"])
                / eval_df.shape[0],
                random_state=settings["hyperparameters"]["seed"],
            )
        )

    @flow
    def predict_flow():
        """Prediction flow"""

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
        predictions = run_predict(masked_data_module)

        verb_mapping = reduce_verbs(
            pretrained_model_name_or_path=settings["pretrained_model_name_or_path"],
            british_spellings_json_path=settings["british_spellings_json_path"],
            predictions=predictions,
            opts=settings["hyperparameters"],
        )
        sub_df = prepare_substitute_sentences(
            df=masked_df,
            pretrained_model_name_or_path=settings["pretrained_model_name_or_path"],
            british_spellings_json_path=settings["british_spellings_json_path"],
            verb_mapping=verb_mapping,
        )
        entailment_output = check_entailment(
            df=sample_eval(sub_df),
            batch_size=settings["hyperparameters"]["nli_model_batch_size"],
            nli_model=settings["allen_nli_model"],
        )
        save_predictions_to_file(
            masked_df, entailment_output, verb_mapping, predictions
        )

    predict_flow()


if __name__ == "__main__":
    predict_fn()
