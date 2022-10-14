import unittest

from transformers import AutoTokenizer

from config import settings
from verb_cluster.tasks import reduce_verbs


class VerbSATReductionTest(unittest.TestCase):
    def test_verb_reduction(self):
        pretrained_model_name_or_path = settings["pretrained_model_name_or_path"]
        british_spellings_json_path = settings["british_spellings_json_path"]
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

        predictions = {
            0: {
                "original_verb": "induce",
                "substitute_verb": (
                    [0.23, 0.20, 0.04, 0.01, 0.49, 0.0],
                    tokenizer.encode(
                        " ".join(
                            [
                                "express",
                                "stimulate",
                                "activate",
                                "enhance",
                                "increase",
                                "induce",
                            ]
                        )
                    )[1:-1],
                ),
            },
            1: {
                "original_verb": "enhance",
                "substitute_verb": (
                    [0.23, 0.20, 0.04, 0.00, 0.49],
                    tokenizer.encode(
                        " ".join(["good", "improve", "activate", "enhance", "increase"])
                    )[1:-1],
                ),
            },
        }

        verb_mapping = reduce_verbs.fn(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            predictions=predictions,
            british_spellings_json_path=british_spellings_json_path,
            opts={"verb_frequency_threshold": 1},
        )
        assert verb_mapping["induce"] == "increase"
        assert verb_mapping["enhance"] == "increase"

        verb_mapping = reduce_verbs.fn(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            predictions=predictions,
            british_spellings_json_path=british_spellings_json_path,
        )

        assert len(verb_mapping) == 0

    def test_verb_reduction_higher_threshold(self):
        pretrained_model_name_or_path = settings["pretrained_model_name_or_path"]
        british_spellings_json_path = settings["british_spellings_json_path"]
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

        predictions = {
            0: {
                "original_verb": "induce",
                "substitute_verb": (
                    [0.23, 0.20, 0.04, 0.41, 0.09, 0.0],
                    tokenizer.encode(
                        " ".join(
                            [
                                "express",
                                "stimulate",
                                "activate",
                                "enhance",
                                "increase",
                                "induce",
                            ]
                        )
                    )[1:-1],
                ),
            },
            1: {
                "original_verb": "improve",
                "substitute_verb": (
                    [0.23, 0.20, 0.04, 0.00, 0.09],
                    tokenizer.encode(
                        " ".join(["good", "improve", "activate", "enhance", "increase"])
                    )[1:-1],
                ),
            },
        }

        verb_mapping = reduce_verbs.fn(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            predictions=predictions,
            british_spellings_json_path=british_spellings_json_path,
            opts={"verb_frequency_threshold": 1, "sub_verb_score_threshold": 0.6},
        )
        assert len(set(verb_mapping.values())) == 2

        verb_mapping = reduce_verbs.fn(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            predictions=predictions,
            british_spellings_json_path=british_spellings_json_path,
            opts={"verb_frequency_threshold": 1, "sub_verb_score_threshold": 0.05},
        )
        assert len(set(verb_mapping.values())) == 1

    def test_verb_reduction_british_english(self):
        pretrained_model_name_or_path = settings["pretrained_model_name_or_path"]
        british_spellings_json_path = settings["british_spellings_json_path"]
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

        predictions = {
            0: {
                "original_verb": "optimise",
                "substitute_verb": (
                    [0.23, 0.20, 0.04, 0.41, 0.5],
                    tokenizer.encode(
                        " ".join(
                            [
                                "optimize",
                                "stimulate",
                                "activate",
                                "enhance",
                                "agonise",
                            ]
                        )
                    )[1:-1],
                ),
            },
            1: {
                "original_verb": "improve",
                "substitute_verb": (
                    [0.23, 0.20, 0.04, 0.5],
                    tokenizer.encode(
                        " ".join(["good", "improve", "activate", "agonise"])
                    )[1:-1],
                ),
            },
        }

        verb_mapping = reduce_verbs.fn(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            predictions=predictions,
            british_spellings_json_path=british_spellings_json_path,
            opts={"verb_frequency_threshold": 1, "sub_verb_score_threshold": 0.1},
        )
        assert "optimize" in verb_mapping
