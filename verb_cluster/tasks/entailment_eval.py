from typing import Any, Dict, List

import pandas as pd
import torch
from allennlp_models.pretrained import load_predictor
from loguru import logger
from prefect import task
from tqdm import tqdm


def chunker(seq: pd.DataFrame, size: int) -> List:
    """Chunks the datafram into size bits enabling batch prediction using batch predictor

    Args:
        seq (pd.DataFrame): dataframe to chunk
        size (int): Chunk size

    Returns:
        List: List of size or less of dataframe
    """
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


@task
def check_entailment(
    df: pd.DataFrame,
    nli_model: str = "pair-classification-roberta-mnli",
    batch_size: int = 32,
    cuda_device: int = 0,
) -> Dict[str, Any]:
    """Check entailment across the original sentence and substituted sentence

    Args:
        df (pd.DataFrame): dataframe to utlize
        nli_model (str) : AllenNLP NLI model name
        batch_size (int): batch size for the batched prediction
        cuda_device (int): cuda device id used to run the model on cuda

    Returns:
        Dict[str, Any]: Returns data with rows where the original sentence
         does not entail the substitute sentence. Along with the precentage
         of sentences that entails along with the average confidence score
    """

    logger.info(f"Evaluation data count: {len(df)}")

    if len(df) == 0:
        logger.warning("DataFrame is empty")
        return {
            "entailment_percentage": 0,
            "entailment_confidence_score": 0,
            "missed_df_data": pd.DataFrame({}),
        }

    if torch.cuda.is_available() and cuda_device != -1:
        logger.info("Cuda device available. Running NLI model on the CUDA")
        predictor = load_predictor(nli_model, cuda_device=cuda_device)
    else:
        logger.info("Cuda device not available. Running NLI model on the CPU")
        predictor = load_predictor(nli_model)

    entailment_score, entailment_count = 0, 0
    missed_df_data = []
    for data in tqdm(
        chunker(df, batch_size),
        total=int(df.shape[0] / batch_size),
        desc="Calculating entailement",
    ):
        all_probs = predictor.predict_batch_json(
            [
                dict(
                    premise=row["sentence"],
                    hypothesis=row["substitute_sent"],
                )
                for _, row in data.iterrows()
            ]
        )
        for index, pred in enumerate(all_probs):
            probs = pred["probs"]
            entailment_score += probs[0]
            if probs[0] < probs[1] or probs[0] < probs[2]:
                missed_df_data.append(
                    {
                        **data.iloc[index].to_dict(),
                        "entailement": probs[0],
                        "neutral": probs[1],
                        "contradiction": probs[2],
                    }
                )
            else:
                entailment_count += 1
    logger.success(
        f"Entailment percentage: {entailment_count/ len(df)},"
        + f"Entailment confidence_score: {entailment_score/len(df)}"
    )
    return {
        "missed_df_data": pd.DataFrame(missed_df_data),
        "entailment_percentage": entailment_count / len(df),
        "entailment_confidence_score": entailment_score / len(df),
    }
