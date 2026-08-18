"""
DATA VALIDATION
===============

Validates the Telco Customer Churn dataset before model training.

Implements the same validation suite previously written against the
Great Expectations API. GE 1.x does not support Python 3.13, so these
checks are implemented directly with pandas — same logic, no dependency
on an incompatible library.

Each check is named after the equivalent GE expectation type so the
comparison is transparent in code review.

Validation categories:
  1. Schema / column existence
  2. Business logic (allowed categorical values)
  3. Numeric ranges
  4. Null checks
  5. Cross-column consistency

Usage:
    from src.utils.validate_data import validate_telco_data
    is_valid, failed = validate_telco_data(df)
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Expectation definitions
# ---------------------------------------------------------------------------
# Each entry is (check_name, callable -> bool, description).
# The callable receives the DataFrame and returns True when the check PASSES.


def _check_column_exists(df: pd.DataFrame, col: str) -> tuple[bool, str]:
    """expect_column_to_exist"""
    passed = col in df.columns
    return passed, f"expect_column_to_exist: '{col}'"


def _check_not_null(df: pd.DataFrame, col: str, mostly: float = 1.0) -> tuple[bool, str]:
    """expect_column_values_to_not_be_null"""
    if col not in df.columns:
        return False, f"expect_column_values_to_not_be_null: '{col}' (column missing)"
    null_rate = df[col].isna().mean()
    passed = null_rate <= (1.0 - mostly)
    return passed, f"expect_column_values_to_not_be_null: '{col}' (null_rate={null_rate:.3f})"


def _check_in_set(
    df: pd.DataFrame, col: str, valid_values: set, mostly: float = 1.0
) -> tuple[bool, str]:
    """expect_column_values_to_be_in_set"""
    if col not in df.columns:
        return False, f"expect_column_values_to_be_in_set: '{col}' (column missing)"
    invalid_mask = ~df[col].isin(valid_values) & df[col].notna()
    invalid_rate = invalid_mask.mean()
    passed = invalid_rate <= (1.0 - mostly)
    return passed, (
        f"expect_column_values_to_be_in_set: '{col}' "
        f"(invalid_rate={invalid_rate:.3f}, allowed={sorted(valid_values)})"
    )


def _check_between(
    df: pd.DataFrame,
    col: str,
    min_value: float = None,
    max_value: float = None,
    mostly: float = 1.0,
) -> tuple[bool, str]:
    """expect_column_values_to_be_between"""
    if col not in df.columns:
        return False, f"expect_column_values_to_be_between: '{col}' (column missing)"
    numeric = pd.to_numeric(df[col], errors="coerce")
    out_of_range = pd.Series(False, index=df.index)
    if min_value is not None:
        out_of_range |= numeric < min_value
    if max_value is not None:
        out_of_range |= numeric > max_value
    out_of_range &= numeric.notna()  # ignore nulls — separate null check
    violation_rate = out_of_range.mean()
    passed = violation_rate <= (1.0 - mostly)
    return passed, (
        f"expect_column_values_to_be_between: '{col}' "
        f"[{min_value}, {max_value}] (violation_rate={violation_rate:.3f})"
    )


def _check_pair_A_gte_B(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    mostly: float = 0.95,
) -> tuple[bool, str]:
    """expect_column_pair_values_A_to_be_greater_than_B (or_equal=True)"""
    if col_a not in df.columns or col_b not in df.columns:
        return False, "expect_column_pair_A_gte_B: missing column(s)"
    a = pd.to_numeric(df[col_a], errors="coerce")
    b = pd.to_numeric(df[col_b], errors="coerce")
    both_present = a.notna() & b.notna()
    violated = both_present & (a < b)
    violation_rate = violated.mean()
    passed = violation_rate <= (1.0 - mostly)
    return passed, (
        f"expect_column_pair_A_gte_B: '{col_a}' >= '{col_b}' "
        f"(violation_rate={violation_rate:.3f}, threshold={1.0 - mostly:.2f})"
    )


# ---------------------------------------------------------------------------
# Public validation function
# ---------------------------------------------------------------------------


def validate_telco_data(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Run the full validation suite on the Telco Customer Churn dataset.

    Implements the same checks as the previous Great Expectations suite,
    rewritten in native pandas for Python 3.13 compatibility.

    Args:
        df: Raw DataFrame loaded directly from the CSV (before preprocessing).

    Returns:
        (is_valid, failed_checks)
        - is_valid:      True if all checks passed.
        - failed_checks: List of descriptive strings for each failed check.
                         Empty list when all checks pass.
    """
    print("[*] Starting data validation...")

    results: list[tuple[bool, str]] = []

    # --- 1. Schema: required columns ---
    print("   [+] Schema checks...")
    required_cols = [
        "customerID",
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "InternetService",
        "Contract",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]
    for col in required_cols:
        results.append(_check_column_exists(df, col))

    # customerID must not be null (business requirement: every record needs an ID)
    results.append(_check_not_null(df, "customerID"))

    # --- 2. Business logic: allowed categorical values ---
    print("   [+] Business logic checks...")
    results.append(_check_in_set(df, "gender", {"Male", "Female"}))
    results.append(_check_in_set(df, "Partner", {"Yes", "No"}))
    results.append(_check_in_set(df, "Dependents", {"Yes", "No"}))
    results.append(_check_in_set(df, "PhoneService", {"Yes", "No"}))
    results.append(_check_in_set(df, "InternetService", {"DSL", "Fiber optic", "No"}))
    results.append(_check_in_set(df, "Contract", {"Month-to-month", "One year", "Two year"}))

    # --- 3. Numeric ranges ---
    print("   [+] Numeric range checks...")
    results.append(_check_between(df, "tenure", min_value=0, max_value=120))
    results.append(_check_between(df, "MonthlyCharges", min_value=0, max_value=200))
    results.append(_check_between(df, "TotalCharges", min_value=0))

    # --- 4. Null checks on critical numeric features ---
    results.append(_check_not_null(df, "tenure"))
    results.append(_check_not_null(df, "MonthlyCharges"))

    # --- 5. Cross-column consistency ---
    # TotalCharges >= MonthlyCharges for >= 95% of rows
    # (new customers in month 1 may have equal values)
    print("   [+] Cross-column consistency checks...")
    results.append(_check_pair_A_gte_B(df, "TotalCharges", "MonthlyCharges", mostly=0.95))

    # --- Summarise ---
    passed = [r for r in results if r[0]]
    failed = [r for r in results if not r[0]]
    failed_descriptions = [desc for _, desc in failed]

    total = len(results)
    n_pass = len(passed)
    n_fail = len(failed)
    is_valid = n_fail == 0

    if is_valid:
        print(f"[OK] Data validation PASSED: {n_pass}/{total} checks successful")
    else:
        print(f"[FAIL] Data validation FAILED: {n_fail}/{total} checks failed")
        for desc in failed_descriptions:
            print(f"   x {desc}")

    return is_valid, failed_descriptions
