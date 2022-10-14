import collections
import json
from collections import defaultdict
from functools import cache
from typing import Dict

import numpy as np
import spacy
from loguru import logger
from nltk.corpus import stopwords, wordnet
from prefect import task
from pysat.examples.hitman import Hitman
from tqdm import tqdm
from transformers import AutoTokenizer


@task
def reduce_verbs(
    pretrained_model_name_or_path: str,
    british_spellings_json_path: str,
    predictions: Dict,
    spacy_model: str = "en_core_web_sm",
    opts: Dict = {},
) -> Dict[str, str]:
    """Given the predictions from the self-supervised model
        reduce the verb cardinality using HittingSet SAT Solver

    Args:
        pretrained_model_name_or_path (str): Pretrained transformer model
            used to encode the sentence
        british_spellings_json_path (str): Path to the JSON file that maps British
            verbs to American verbs
        predictions (Dict): predictions from the self-supervised model
        spacy_model (str): Spacy model used to lemmatize the verbs
        opts (Dict): Hyperparameters

    Returns:
        Dict[str, str]: Returns the verb mapping that maps source verbs to
            substitute verbs
    """
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    nlp = spacy.load(spacy_model)
    stop = set(stopwords.words("english"))

    default_opts = {
        "verb_frequency_threshold": 20,
        "sub_verb_score_threshold": 0.2,
    }

    opts = {**default_opts, **opts}
    verb_frequency_threshold = opts.get("verb_frequency_threshold")
    logger.info(f"Reduction hyperparameters: {opts}")

    with open(british_spellings_json_path) as f:
        british_to_american_dict = json.load(f)

    process_verb = cache(lambda verb: nlp(verb)[0])
    tokenize = cache(lambda indice: tokenizer.decode([indice]))
    convert_british_to_american = (
        lambda verb: british_to_american_dict[verb]
        if verb in british_to_american_dict
        else verb
    )

    @cache
    def get_all_ant(verb):
        antonyms = []
        for syn in wordnet.synsets(verb):
            for lm in syn.lemmas():
                if lm.antonyms():
                    antonyms.append(lm.antonyms()[0].name())
        return set(antonyms)

    verb_substitute_index = {}
    missed_sent = 0
    ignored_verb_sent = 0

    original_verbs_list = map(lambda d: d["original_verb"], predictions.values())
    counter = collections.Counter(original_verbs_list)
    selected_original_verbs = set(
        [
            convert_british_to_american(v)
            for v, k in counter.items()
            if k >= verb_frequency_threshold
        ]
    )
    logger.info(f"Original verb count: {len(counter)}")
    logger.info(f"Filtered verb count: {len(selected_original_verbs)}")

    for pred_index, pred in tqdm(
        predictions.items(), "Building verb replacement index"
    ):
        original_verb = convert_british_to_american(pred["original_verb"])
        scores, verb_indices = pred["substitute_verb"]

        if original_verb not in selected_original_verbs:
            ignored_verb_sent += 1
            continue
        verbs = [tokenize(ind) for ind in verb_indices]
        lemmatized_verbs = defaultdict(lambda: 0)
        for index, verb in enumerate(verbs):
            doc = process_verb(verb)
            lemma, pos = doc.lemma_, doc.pos_
            lemma = convert_british_to_american(lemma)
            if lemma not in stop and "VERB" == pos:
                lemmatized_verbs[lemma] = max(lemmatized_verbs[lemma], scores[index])

        lemmatized_verbs = dict(lemmatized_verbs)
        if original_verb not in lemmatized_verbs:
            missed_sent += 1
            continue
        # TODO: Explore other ways of normalizing the scores
        lemmatized_verbs[original_verb] = 0
        lem_scores = np.array(list(lemmatized_verbs.values()))

        if (lem_scores.max() - lem_scores.min()) > 0:
            lem_scores = (lem_scores - lem_scores.min()) / (
                lem_scores.max() - lem_scores.min()
            )
        else:
            missed_sent += 1
            continue
        for i, v in enumerate(lemmatized_verbs):
            lemmatized_verbs[v] = lem_scores[i]

        verb_substitute_index[pred_index] = {
            "original_verb": original_verb,
            "substitute_verbs": lemmatized_verbs,
        }

    logger.info(f"Ignore sentences based on verb_threshold: {ignored_verb_sent}")
    logger.info(f"Ignored sentences based on prediction: {missed_sent}")

    verb_pred_score = defaultdict(lambda: defaultdict(lambda: (0, 0)))
    for pred_index, pred in tqdm(verb_substitute_index.items(), "Compiling verb index"):
        original_verb = pred["original_verb"]
        subs = pred["substitute_verbs"]

        for index, (sub, sub_score) in enumerate(subs.items()):
            verb_pred_score[original_verb][sub] = (
                verb_pred_score[original_verb][sub][0] + sub_score,
                verb_pred_score[original_verb][sub][1] + 1,
            )

    updated_verb_pred_score = defaultdict(lambda: {})
    combined_verb_score = defaultdict(lambda: (0, 0))
    for s_v, targets in tqdm(verb_pred_score.items()):
        for t_v, (sc, count) in targets.items():
            updated_verb_pred_score[s_v][t_v] = sc / counter[s_v]
            combined_verb_score[t_v] = (
                combined_verb_score[t_v][0] + (sc / counter[s_v]),
                combined_verb_score[t_v][1] + 1,
            )

    combined_verb_score = {
        v: sc / count for v, (sc, count) in combined_verb_score.items()
    }

    verb_cover_set = defaultdict(lambda: set())
    for s_v, targets in tqdm(updated_verb_pred_score.items()):
        for t_v, score in targets.items():
            if (
                score > opts.get("sub_verb_score_threshold")
                and s_v not in get_all_ant(t_v)
            ) or s_v == t_v:
                verb_cover_set[s_v].add(t_v)

    h = Hitman(solver="m22", htype="lbx")
    for t_v, subs in tqdm(verb_cover_set.items()):
        weights = {v: 1 / combined_verb_score[v] for v in subs}
        h.hit(subs, weights)

    reduced_verbs = h.get()

    verb_mapping = {}
    for s_v, targets in tqdm(verb_cover_set.items(), "Building Verb Lookup table"):
        if s_v in reduced_verbs:
            verb_mapping[s_v] = s_v
        else:
            sel = {
                k: updated_verb_pred_score[s_v][k]
                for k in targets
                if k in reduced_verbs and k in updated_verb_pred_score[s_v]
            }
            if len(sel) > 0:
                verb_mapping[s_v] = max(sel, key=sel.get)
            else:
                verb_mapping[s_v] = s_v

    reduced_set = set(list(verb_mapping.values()))

    logger.success(f"Reduced verb set size: {len(reduced_set)}")

    return verb_mapping
