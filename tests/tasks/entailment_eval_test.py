import unittest

import pandas as pd

from verb_cluster.tasks import check_entailment


class EntailmentEvalTest(unittest.TestCase):
    def test_entailment_score(self):
        df = pd.DataFrame(
            {
                "substitute_sent": ["Gene1 enhance Gene2 expression"],
                "sentence": ["Gene1 increases Gene2 expression"],
            }
        )
        entailment_results = check_entailment.fn(df=df)
        assert abs(entailment_results["entailment_percentage"] - 1.0) < 1e-5

    def test_contradiction_score(self):
        df = pd.DataFrame(
            {
                "substitute_sent": ["Gene1 enhance Gene2 expression"],
                "sentence": ["Gene1 decreases Gene2 expression"],
            }
        )
        entailment_results = check_entailment.fn(df=df)
        assert abs(entailment_results["entailment_percentage"] - 0.0) < 1e-5

    def test_empty_entail_test(self):
        empty_df = pd.DataFrame({})

        entailment_results = check_entailment.fn(df=empty_df)
        assert entailment_results["entailment_percentage"] == 0
