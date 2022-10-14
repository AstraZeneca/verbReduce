import glob
import os
import shutil
import unittest

import pandas as pd

from config import settings
from verb_cluster.flows import predict_fn, train_fn

os.environ["ENV_FOR_DYNACONF"] = "test"


class FlowTest(unittest.TestCase):
    def test_predict_flow(self):
        prediction_path = settings["path"]["prediction_path"]
        if os.path.exists(prediction_path) and os.path.isdir(prediction_path):
            shutil.rmtree(prediction_path)
        predict_fn()
        lut_path = os.path.join(prediction_path, "lut.csv")
        sub_path = os.path.join(prediction_path, "subs.csv")
        assert os.path.exists(lut_path)
        assert os.path.exists(sub_path)

        lut_df = pd.read_csv(lut_path)
        sub_df = pd.read_csv(sub_path)

        assert lut_df.shape[1] == 3
        assert sub_df.shape[0] == 26

    def test_train_flow(self):
        checkpoint_path = settings["path"]["checkpoint_path"]
        if os.path.exists(checkpoint_path) and os.path.isdir(checkpoint_path):
            shutil.rmtree(checkpoint_path)
        train_fn()
        ckpt_files = glob.glob(f"{checkpoint_path}/*.ckpt")
        assert len(ckpt_files) > 0
