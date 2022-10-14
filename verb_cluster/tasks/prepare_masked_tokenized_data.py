from typing import Dict, List, Tuple

import pandas as pd
import torch
from loguru import logger
from prefect import task
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer

tqdm.pandas()


def _find_sub_list(sl: List, lst: List) -> List[Tuple[int, int]]:
    """Find sublist position within list

    Args:
        sl (list): sublist to search
        lst: (list): list to search sublist for

    Returns:
        List[Tuple[int, int]]: returns list of tuples with
            start position and end position of sublist within list
    """
    results = []
    sll = len(sl)
    for ind in (i for i, e in enumerate(lst) if e == sl[0]):
        if lst[ind : ind + sll] == sl:
            results.append((ind, ind + sll - 1))
    return results


def _process_tokenize(
    index: int,
    sentence: str,
    masked_sent: str,
    verb_form: str,
    tokenizer: AutoTokenizer,
    masked_seq_encoded: str,
    max_seq_length: int,
) -> Tuple[Dict[str, Tensor], Tensor, Tensor]:
    """Tokenize the sentence and masked sentence

    Args:
        index (int): position of text in dataframe
        sentence (str): Original sentence
        masked_sent (str): Masked sentence
        verb_form (str): the form of verb in text
        tokenizer (AutoTokenizer): Tokenizer for tokenizing
        masked_seq_encoded (str): Mask sequence text
        max_seq_length (int): Maximum length of text

    Returns:
        Tuple[Dict[str, Tensor], Tensor, Tensor]: returns the tokenized text
            along with the mask tensor and verb mask tensor
    """
    encoded_sentence_ids = tokenizer(
        sentence,
        masked_sent,
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    verb_ids = tokenizer(verb_form, add_special_tokens=False)["input_ids"]
    try:
        results = _find_sub_list(
            verb_ids, encoded_sentence_ids["input_ids"].numpy().tolist()[0]
        )[0]
        mask_results = _find_sub_list(
            [masked_seq_encoded], encoded_sentence_ids["input_ids"].numpy().tolist()[0]
        )[0]

        verb_mask = torch.zeros_like(encoded_sentence_ids["input_ids"][0])
        verb_mask[results[0] : results[1] + 1] = 1

        mask_tensor = torch.zeros_like(encoded_sentence_ids["input_ids"][0])
        mask_tensor[mask_results[0]] = 1

        # TODO: Currenlty ignores verbs with multiple token verbs
        if len(verb_ids) > 1:
            return None, None, None, None

        return (
            {k: v[0] for k, v in encoded_sentence_ids.items()},
            mask_tensor,
            verb_mask,
            verb_ids,
        )
    except IndexError:
        logger.error(f"Index error at: {index}")
        return None, None, None, None


@task
def prepare_masked_tokenized_data(
    df: pd.DataFrame,
    pretrained_model_name_or_path: str = "bert-large-uncased-whole-word-masking",
    max_seq_length: int = 256,
) -> pd.DataFrame:
    """Tokenizes the original sentence and masked sentence

    Args:
        df (pd.DataFrame): Dataframe with original sentence and masked sentence
        pretrained_model_name_or_path (str): Pretrained transformer model for
            training and inference
        max_seq_length (int): Maximum sequence length (original sentence
            and masked sentence combined)
    Returns:
        pd.DataFrame: returns the dataframe with the tokenized text
    """
    logger.info(f"Encoding data frame of shape: {df.shape}")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

    masked_seq_encoded = tokenizer.mask_token_id
    (df["encoded_sentence"], df["mask_index"], df["verb_mask"], df["verb_ids"]) = zip(
        *df.progress_apply(
            lambda row: _process_tokenize(
                row.name,
                row["sentence"],
                row["masked_sent"],
                row["verb_form"],
                tokenizer,
                masked_seq_encoded,
                max_seq_length=max_seq_length,
            ),
            axis=1,
        )
    )
    df = df[df["encoded_sentence"].notna()]
    logger.info(f"DataFrame encoded with shape: {df.shape}")

    return df
