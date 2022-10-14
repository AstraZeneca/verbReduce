import json
from typing import Dict, Tuple

import pandas as pd
from loguru import logger
from prefect import task
from tqdm import tqdm
from transformers import AutoTokenizer

tqdm.pandas()


def _replace_verbs(
    masked_sent: str, verb: str, verb_mapping: Dict[str, str], mask_token: str
) -> Tuple[str, str, str]:
    """Given masked sentence, replace the source verb with substitute verb.

    Args:
        masked_sent (str): Masekd sentence
        verb (str): Lemmatized verb to replace
        verb_mapping (Dict[str:,str]): Dict mapping source verb to substitute verb
        mask_token (str): Mask token of the pretrained transformer model

    Returns:
        Tuple(str, str, str): Returns substituted verb, substituted
            sentence and original verb in American format
    """
    sub_verb = verb_mapping[verb] if verb in verb_mapping else verb
    sub_sent = masked_sent.replace(mask_token, sub_verb)
    return sub_verb, sub_sent, verb


@task
def prepare_substitute_sentences(
    pretrained_model_name_or_path: str,
    df: pd.DataFrame,
    verb_mapping: Dict[str, str],
    british_spellings_json_path: str,
) -> pd.DataFrame:
    """Given data frame, verb mapping that maps from the source verb
    to the target verb and masked sentence, this function creates a
    new columns in the dataframe substitute_sent with the masked verb
    replace with the target verb

    Args:
        pretrained_model_name_or_path (str): Pretrained transformer model
            used to encode the sentence
        df (pd.DataFrame): Dataframe with verb, sentence and masked sentence
        verb_mapping (Dict[str, str]): Verb mapping from source verb to target verb
        british_spellings_json_path (str): Path to the JSON file that maps British
            verbs to American verbs

    Returns:
        pd.DataFrame: returns dataframe with the substituted sentence, substituted verb
        and the original verb in American English form
    """
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    mask_token = tokenizer.mask_token

    with open(british_spellings_json_path) as f:
        british_to_american_dict = json.load(f)

    convert_british_to_american = (
        lambda verb: british_to_american_dict[verb]
        if verb in british_to_american_dict
        else verb
    )

    logger.info("Preparing substituted sentences")

    # If substitute_sent column already exists, then remove them
    # We encouter this case mostly during validation_epoch_end of training
    if "substitute_sent" in df:
        df = df.drop(["substitute_sent", "substitute_verb", "original_verb"], axis=1)
    (df["substitute_verb"], df["substitute_sent"], df["original_verb"],) = zip(
        *df.progress_apply(
            lambda row: _replace_verbs(
                row["masked_sent"],
                convert_british_to_american(row["verb"]),
                verb_mapping,
                mask_token,
            ),
            axis=1,
        ),
    )

    original_verb_count = df["original_verb"].nunique()
    sub_verb_count = df["substitute_verb"].nunique()

    logger.success(f"Number of original_verbs: {original_verb_count}")
    logger.success(f"Number of substitute verbs: {sub_verb_count}")

    return df
