from functools import reduce

import pandas as pd
import pytorch_lightning as pt
import torch
from transformers import AdamW, get_linear_schedule_with_warmup

from verb_cluster.tasks import (check_entailment, prepare_substitute_sentences,
                                reduce_verbs)

from .models import MaskedModel


class MaskedPlModel(pt.LightningModule):
    """Masked Pytorch Ligthning Module

    Please refer to the following link:
    https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html
    to understand further what each function does
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        british_spellings_json_path: str,
        task_name: str = "SelfSupervisedMaskingTask",
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        test_df: pd.DataFrame = None,
        test_count: int = 5000,
        verb_frequency_threshold: int = 20,
        sub_verb_score_threshold: float = 0.2,
        top_k: int = 20,
        **kwargs,
    ):
        """Init function

        Args:
            pretrained_model_name_or_path (str): Pretrained transformer model
                used to encode the sentence
            british_spellings_json_path (str): Path to the JSON file that maps British
                verbs to American verbs
            task_name (str): Name of the task
            learning_rate (float): Training learning rate
            adam_epsilon (float): Adam epsilon for Adam optimizer
            warmup_steps (int): Number of warm up steps
            weight_decay (float): Weight decay regualarizer
            train_batch_size (int): Batch size for training
            eval_batch_size (int): Batch size for eval
            test_df (pd.DataFrame): Testing datafram
            test_count (int): Number of samples to test
            verb_frequency_threshold (int): Cutoff threshold of verbs less than
                the value
            sub_verb_score_threshold (float): Cutoff threshold of verbs to
                the verb replacement
            top_k (int): Top K predictions to be considered for replacement
        """
        super().__init__()

        self.save_hyperparameters(ignore=["test_df", "british_spellings_json_path"])

        self.model = MaskedModel(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )
        self.british_spellings_json_path = british_spellings_json_path
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.test_df = test_df

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits, token_prediction = outputs[:3]

        predictions = {}

        for idx, pos in batch["mask_index"].detach().cpu().nonzero().numpy().tolist():
            score, indices = torch.topk(token_prediction[idx, pos, :], k=20)
            score = score.cpu().numpy().tolist()
            indices = indices.cpu().numpy().tolist()

            predictions[batch["index"][idx].cpu().item()] = {
                "original_verb": batch["verb"][idx],
                "substitute_verb": (score, indices),
            }

        return {"loss": val_loss, "predictions": predictions}

    def predict_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        predictions = reduce(lambda s, d: {**s, **d["predictions"]}, outputs, {})
        verb_mapping = reduce_verbs.fn(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            british_spellings_json_path=self.british_spellings_json_path,
            predictions=predictions,
            opts={
                "verb_frequency_threshold": self.hparams.verb_frequency_threshold,
                "sub_verb_score_threshold": self.hparams.sub_verb_score_threshold,
            },
        )
        sub_df = prepare_substitute_sentences.fn(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            df=self.test_df,
            british_spellings_json_path=self.british_spellings_json_path,
            verb_mapping=verb_mapping,
        )

        sub_verb_count = sub_df["substitute_verb"].nunique()

        sub_df = sub_df.drop_duplicates(subset=["sentence"], keep="first")
        eval_df = sub_df[sub_df["substitute_verb"] != sub_df["original_verb"]]

        if len(eval_df) > 0:
            eval_df = eval_df.groupby("original_verb", group_keys=False).apply(
                lambda x: x.sample(
                    frac=min(eval_df.shape[0], self.hparams.test_count)
                    / eval_df.shape[0],
                    random_state=42,
                )
            )

            entailment_result = check_entailment.fn(eval_df)
            entailment_percentage = entailment_result["entailment_percentage"]
        else:
            entailment_percentage = 0

        loss = torch.stack([x["loss"].detach().cpu() for x in outputs]).mean()
        self.log("sub_verb_count", float(sub_verb_count), prog_bar=True)
        self.log(
            "entailment_percentage",
            entailment_percentage,
            prog_bar=True,
        )
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
