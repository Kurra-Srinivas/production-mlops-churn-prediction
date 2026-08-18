"""
Unit tests for src/pipeline/preprocess.py

Tests verify:
  - Pipeline produces the correct output shape and dtype
  - Feature order is deterministic
  - Unknown categories are handled gracefully (no crash, all-zero row)
  - Pipeline is serialisable / deserialisable with identical output
  - scale_numerics flag works correctly
  - No pd.get_dummies is used anywhere in the module
"""

import inspect
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.pipeline.preprocess import (
    BINARY_COLS,
    NUMERIC_COLS,
    OHE_COLS,
    build_preprocessing_pipeline,
    fit_pipeline,
    get_feature_names,
    load_pipeline,
    save_pipeline,
)


# ---------------------------------------------------------------------------
# Helper: expected feature count
# ---------------------------------------------------------------------------
# Binary cols:  5 cols × 1 output each  = 5
# Numeric cols: 4 cols × 1 output each  = 4
# OHE cols: each col contributes (n_categories - 1) outputs with drop='first'
# Expected OHE columns (from the dataset):
#   MultipleLines:      3 cats → 2 cols
#   InternetService:    3 cats → 2 cols
#   OnlineSecurity:     3 cats → 2 cols
#   OnlineBackup:       3 cats → 2 cols
#   DeviceProtection:   3 cats → 2 cols
#   TechSupport:        3 cats → 2 cols
#   StreamingTV:        3 cats → 2 cols
#   StreamingMovies:    3 cats → 2 cols
#   Contract:           3 cats → 2 cols
#   PaymentMethod:      4 cats → 3 cols
# OHE total: 9×2 + 1×3 = 21
EXPECTED_N_FEATURES = 5 + 4 + 21  # = 30


class TestPipelineOutputShape:

    def test_output_feature_count(self, fitted_pipeline, raw_df):
        """Transformed array must have exactly 30 features."""
        X = raw_df[BINARY_COLS + NUMERIC_COLS + OHE_COLS]
        X_out = fitted_pipeline.transform(X)
        assert X_out.shape[1] == EXPECTED_N_FEATURES, (
            f"Expected {EXPECTED_N_FEATURES} features, got {X_out.shape[1]}"
        )

    def test_output_is_2d(self, fitted_pipeline, raw_df):
        """Output must be a 2-D array."""
        X = raw_df[BINARY_COLS + NUMERIC_COLS + OHE_COLS]
        X_out = fitted_pipeline.transform(X)
        assert X_out.ndim == 2

    def test_output_dtype_is_numeric(self, fitted_pipeline, raw_df):
        """All output columns must be numeric (float64)."""
        X = raw_df[BINARY_COLS + NUMERIC_COLS + OHE_COLS]
        X_out = fitted_pipeline.transform(X)
        assert np.issubdtype(X_out.dtype, np.floating), (
            f"Expected float dtype, got {X_out.dtype}"
        )

    def test_single_row_output_shape(self, fitted_pipeline, sample_customer):
        """Single-row transform must work and return (1, 30) array."""
        df = pd.DataFrame([sample_customer])
        X_out = fitted_pipeline.transform(df)
        assert X_out.shape == (1, EXPECTED_N_FEATURES)

    def test_row_count_preserved(self, fitted_pipeline, raw_df):
        """Number of rows must be preserved by transform."""
        X = raw_df[BINARY_COLS + NUMERIC_COLS + OHE_COLS]
        X_out = fitted_pipeline.transform(X)
        assert X_out.shape[0] == len(raw_df)


class TestFeatureOrder:

    def test_feature_names_length(self, fitted_pipeline):
        """get_feature_names must return exactly 30 names."""
        names = get_feature_names(fitted_pipeline)
        assert len(names) == EXPECTED_N_FEATURES

    def test_feature_names_are_strings(self, fitted_pipeline):
        """All feature names must be strings."""
        names = get_feature_names(fitted_pipeline)
        assert all(isinstance(n, str) for n in names)

    def test_feature_names_deterministic(self, raw_df):
        """Two pipelines fitted on the same data must return the same feature names."""
        feature_cols = BINARY_COLS + NUMERIC_COLS + OHE_COLS
        X = raw_df[feature_cols]

        pipeline_a = build_preprocessing_pipeline()
        pipeline_b = build_preprocessing_pipeline()
        pipeline_a.fit(X)
        pipeline_b.fit(X)

        names_a = get_feature_names(pipeline_a)
        names_b = get_feature_names(pipeline_b)
        assert names_a == names_b

    def test_binary_cols_appear_first(self, fitted_pipeline):
        """Binary columns should appear in the first N positions."""
        names = get_feature_names(fitted_pipeline)
        for col in BINARY_COLS:
            assert col in names, f"Binary column '{col}' not in feature names"
        # First 5 should be the binary columns
        for i, col in enumerate(BINARY_COLS):
            assert names[i] == col, f"Binary col order mismatch at position {i}"


class TestUnknownCategoryHandling:

    def test_unknown_internet_service_no_crash(self, fitted_pipeline, sample_customer):
        """Unknown InternetService value must not raise an exception."""
        customer = sample_customer.copy()
        customer["InternetService"] = "Satellite"  # not in training categories
        df = pd.DataFrame([customer])
        # Should not raise; handle_unknown='ignore' → all-zero OHE row
        X_out = fitted_pipeline.transform(df)
        assert X_out.shape == (1, EXPECTED_N_FEATURES)

    def test_unknown_contract_gives_zero_columns(self, fitted_pipeline, sample_customer):
        """An unknown Contract value should result in zeros for Contract OHE columns."""
        customer = sample_customer.copy()
        customer["Contract"] = "Five year"  # unknown
        df = pd.DataFrame([customer])
        X_out = fitted_pipeline.transform(df)
        names = get_feature_names(fitted_pipeline)
        contract_idxs = [i for i, n in enumerate(names) if n.startswith("Contract")]
        # All contract OHE columns should be 0 for the unknown value
        assert all(X_out[0, idx] == 0.0 for idx in contract_idxs)

    def test_missing_monthly_charges_imputed(self, fitted_pipeline, sample_customer):
        """NaN in MonthlyCharges must be imputed (not cause a crash)."""
        customer = sample_customer.copy()
        customer["MonthlyCharges"] = None
        df = pd.DataFrame([customer])
        X_out = fitted_pipeline.transform(df)
        assert X_out.shape == (1, EXPECTED_N_FEATURES)
        # Imputed value should be finite
        names = get_feature_names(fitted_pipeline)
        mc_idx = names.index("MonthlyCharges")
        assert np.isfinite(X_out[0, mc_idx])


class TestSerialisation:

    def test_save_and_load_identical_output(self, fitted_pipeline, raw_df):
        """Save → load must produce numerically identical transform output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "pipeline.pkl")
            save_pipeline(fitted_pipeline, path)
            loaded = load_pipeline(path)

        X = raw_df[BINARY_COLS + NUMERIC_COLS + OHE_COLS]
        out_original = fitted_pipeline.transform(X)
        out_loaded   = loaded.transform(X)
        np.testing.assert_array_equal(out_original, out_loaded)

    def test_load_missing_file_raises(self):
        """load_pipeline must raise FileNotFoundError for a non-existent path."""
        with pytest.raises(FileNotFoundError):
            load_pipeline("/nonexistent/path/pipeline.pkl")


class TestScaleNumerics:

    def test_scaled_pipeline_output_shape(self, fitted_pipeline_scaled, raw_df):
        """Scaled pipeline must produce the same shape as the unscaled pipeline."""
        X = raw_df[BINARY_COLS + NUMERIC_COLS + OHE_COLS]
        X_out = fitted_pipeline_scaled.transform(X)
        assert X_out.shape[1] == EXPECTED_N_FEATURES

    def test_scaled_pipeline_numerics_have_zero_mean(self, fitted_pipeline_scaled, raw_df):
        """Numeric columns in the scaled pipeline should have approximately zero mean."""
        X = raw_df[BINARY_COLS + NUMERIC_COLS + OHE_COLS]
        X_out = fitted_pipeline_scaled.transform(X)
        names = get_feature_names(fitted_pipeline_scaled)
        for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
            if col in names:
                idx = names.index(col)
                col_mean = X_out[:, idx].mean()
                assert abs(col_mean) < 1e-6, f"Scaled '{col}' mean not ~0: {col_mean}"


class TestNoPandasGetDummies:

    def test_no_get_dummies_in_preprocess_module(self):
        """pd.get_dummies must not be CALLED in the preprocessing pipeline module."""
        import src.pipeline.preprocess as mod
        source = inspect.getsource(mod)
        # Strip comment lines and docstring-only lines, then check for actual calls
        code_lines = [
            line for line in source.splitlines()
            if not line.strip().startswith('#') and not line.strip().startswith('"""')
            and not line.strip().startswith("'")
        ]
        code_only = '\n'.join(code_lines)
        assert '.get_dummies(' not in code_only, (
            "pd.get_dummies() called in src/pipeline/preprocess.py — "
            "use OneHotEncoder instead"
        )

    def test_no_get_dummies_in_inference_module(self):
        """pd.get_dummies must not be CALLED in the inference module source."""
        # Read source directly from file — avoids importing the module,
        # which would trigger eager model/pipeline loading at test time.
        import os
        inference_path = os.path.join("src", "serving", "inference.py")
        with open(inference_path, encoding="utf-8") as f:
            source = f.read()
        # Strip comment lines and check for actual function calls only
        code_lines = [
            line for line in source.splitlines()
            if not line.strip().startswith('#') and not line.strip().startswith('"""')
            and not line.strip().startswith("'")
        ]
        code_only = '\n'.join(code_lines)
        assert '.get_dummies(' not in code_only, (
            "pd.get_dummies() called in src/serving/inference.py — "
            "use pipeline.transform() instead"
        )
