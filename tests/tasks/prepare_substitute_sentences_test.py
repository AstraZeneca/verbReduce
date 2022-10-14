import unittest

import pandas as pd

from config import settings
from verb_cluster.tasks import prepare_substitute_sentences


class PrepareSubstituteSentencesTest(unittest.TestCase):
    def test_sentence_substitution(self):
        pretrained_model_name_or_path = settings["pretrained_model_name_or_path"]
        british_spellings_json_path = settings["british_spellings_json_path"]
        test_df = pd.DataFrame(
            {
                "masked_sent": [
                    "This is a [MASK] sentence",
                    "This is a [MASK] sentence with no verb mapped",
                ],
                "verb": ["verb1", "verb3"],
            }
        )
        verb_mapping = {"verb1": "verb2"}

        subs_df = prepare_substitute_sentences.fn(
            df=test_df,
            british_spellings_json_path=british_spellings_json_path,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            verb_mapping=verb_mapping,
        )

        test_data = subs_df.iloc[0]
        assert "substitute_sent" in test_data
        assert test_data["substitute_sent"] == "This is a verb2 sentence"
        test_data2 = subs_df.iloc[1]
        assert (
            test_data2["substitute_sent"]
            == "This is a verb3 sentence with no verb mapped"
        )
