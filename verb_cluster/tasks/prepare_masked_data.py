from typing import Optional, Tuple

import pandas as pd
from loguru import logger
from prefect import task
from tqdm import tqdm
from transformers import AutoTokenizer

tqdm.pandas()


def _mask_sent(
    sent: str,
    verb: str,
    verb_start: int,
    verb_end: int,
    mask_seq: str = "[MASK]",
) -> Tuple[str, str]:
    """Given sentence, verb and position.
    Generate sentence with verb masked with mask_seq

    Args:
        sent (str): sentence with verb
        verb (str): lemmatized verb to mask
        verb_start (int): start position of verb
        verb_end (int): end position of verb
        mask_seq (str): MASK token string for replacing verb with

    Returns:
        Tuple(str, str): returns masked sentence and the verb form found in sentence
    """
    verb = verb.strip().lstrip()
    # NOTE: We are ignoring verbs with more than one words at the moment.
    if len(verb.split()) > 1:
        return None, None
    if verb_start < 0 or verb_start > len(sent) or verb_end < 0 or verb_end > len(sent):
        logger.error("Verb range not found within the sentence")
        return None, None, None
    verb_form = sent[verb_start:verb_end]
    masked_sent = f"{sent[:verb_start]}{mask_seq}{sent[verb_end:]}"
    return masked_sent, verb_form


@task
def prepare_masked_data(
    data_path: str,
    pretrained_model_name_or_path: str,
    verb_frequency_threshold: int,
    sample_size: Optional[float] = None,
) -> pd.DataFrame:
    """Prepare the masked sentence

    Args:
        data_path (pd.DataFrame): path to the data file with verb,sentence,
            verbSeedStart,verbSeedEnd
        pretrained_model_name_or_path (str): Pretrained transformer model for
            training and inference
        verb_frequency_threshold (int): Remove verbs from the dataframe below this
            threshold
        sample_size (Optional[float]): if sample_size is provided, the dataframe is
            sampled proportionately

    Returns:
        pd.DataFrame: returns dataframe with masked sentence
    """
    logger.info(f"Preparing masked data from {data_path}")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    mask_seq = tokenizer.mask_token
    df = pd.read_csv(data_path, error_bad_lines=False, sep="|")

    logger.info(f"Shape of original dataframe: {df.shape}")
    # Remove duplicates
    df = df.drop_duplicates(subset=["sentence", "verb"], keep="first")
    logger.info(f"Shape of dataframe after removing duplicates: {df.shape}")
    logger.info(f"Removing verbs with frequency less than: {verb_frequency_threshold}")
    # Remove rows with verbs occurring less than verb_frequency_threshold
    df = df[df.groupby("verb")["verb"].transform("count").ge(verb_frequency_threshold)]
    logger.info(f"After removing less frequency verbs, new df size: {df.shape}")

    # In some cases where the dataframe size is really large,
    # it might be challenging to train on the entire dataset.
    # In these cases, the dataframe size is reduced via sampling
    if sample_size is not None:
        df = df.groupby("verb", group_keys=False).apply(
            lambda x: x.sample(frac=sample_size)
        )
        logger.info(f"Sample size provided: {sample_size}. New df size: {df.shape}")

    df["masked_sent"], df["verb_form"] = zip(
        *df.progress_apply(
            lambda row: _mask_sent(
                row["sentence"],
                row["verb"],
                int(row["verbSeedStart"]),
                int(row["verbSeedEnd"]),
                mask_seq=mask_seq,
            ),
            axis=1,
        )
    )
    df = df[df["masked_sent"].notna()]
    logger.info(f"Shape of processed dataframe: {df.shape}")
    return df
