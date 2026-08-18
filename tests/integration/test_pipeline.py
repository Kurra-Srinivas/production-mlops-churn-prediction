"""
Integration tests for the end-to-end training pipeline.
"""

import os
import tempfile

from scripts.run_pipeline import main as run_pipeline_main


class DummyArgs:
    def __init__(self, input_path, mlflow_uri):
        self.input = input_path
        self.target = "Churn"
        self.threshold = 0.35
        self.test_size = 0.2
        self.tune = False
        self.n_trials = 2
        self.experiment = "Test_Experiment"
        self.mlflow_uri = mlflow_uri


def test_pipeline_end_to_end(raw_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv = os.path.join(tmpdir, "test_raw.csv")
        mlflow_dir = os.path.join(tmpdir, "mlruns")
        raw_df.to_csv(input_csv, index=False)

        mlflow_uri = f"file:///{mlflow_dir.replace(os.sep, '/')}"
        args = DummyArgs(input_csv, mlflow_uri)

        # Run pipeline
        run_pipeline_main(args)

        # Assert artifacts exist
        assert os.path.exists("artifacts/pipeline.pkl")
        assert os.path.exists("artifacts/feature_columns.txt")
