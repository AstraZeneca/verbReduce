import multiprocessing as mp
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from loguru import logger
from torch.utils.data import DataLoader, Dataset


class MaskedDataset(Dataset):
    """Torch Dataset for the  masked data"""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        input_ids = data["encoded_sentence"]["input_ids"]
        attention_mask = data["encoded_sentence"]["attention_mask"]
        token_type_ids = data["encoded_sentence"]["token_type_ids"]

        mask_pos = data["mask_index"].nonzero()[0]
        verb_id = data["verb_ids"][0]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "verb_mask": data["verb_mask"],
            "mask_index": data["mask_index"],
            "index": idx,
            "verb_form": data["verb_form"],
            "verb": data["verb"],
            "mask_pos": mask_pos,
            "first_verb_token": verb_id,
            "verb_ids": data["verb_ids"],
        }


class MaskedDataModule(pl.LightningDataModule):
    """Pytorch lightning datamodule that contains the train,
    test and predict datasets"""

    def __init__(
        self,
        df: pd.DataFrame,
        test_split: float = 0.2,
        seed: int = 42,
        batch_size: int = 128,
        num_workers: int = 8,
    ):
        """Init function

        Args:
            df (pd.DataFrame): Dataframe with tokenized data
            test_split (float): Test split ration
            seed (int): Random sampling seed
            batch_size (int): Batch size of datasets
            num_workers (int): Num of workers for the datasets
        """
        super().__init__()
        self.df = df
        self.test_split = test_split
        self.seed = seed
        self.batch_size = batch_size
        np.random.seed(seed)

        if num_workers == -1:
            self.num_workers = mp.cpu_count()
        else:
            self.num_workers = num_workers

        self.train_df = self.df.sample(frac=1 - self.test_split, random_state=self.seed)
        self.test_df = self.df.drop(self.train_df.index)

    def setup(self, stage: Optional[str] = None):
        """Setup train, test and predict datasets

        Args:
            stage (Optional[str]): Stage of training, this parameter is not used
                in our code
        """
        logger.info("Setting up MaskedData Module")
        logger.info(f"Train len: {len(self.train_df)} Test len: {len(self.test_df)}")
        self.train = MaskedDataset(self.train_df)
        self.test = MaskedDataset(self.test_df)
        self.predict = MaskedDataset(self.df)

    def train_dataloader(self) -> DataLoader:
        """Returns the train dataloader

        Args:

        Returns:
            DataLoader: Returns the train dataloader
        """
        return DataLoader(
            self.train, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        """Returns the test dataloader

        Args:

        Returns:
            DataLoader: Returns the test dataloader
        """
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        """Returns the validation dataloader

        Args:

        Returns:
            DataLoader: Returns the validation dataloader
        """
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        """Returns the predict dataloader

        Args:

        Returns:
            DataLoader: Returns the predict dataloader
        """
        return DataLoader(
            self.predict, batch_size=self.batch_size, num_workers=self.num_workers
        )
