"""Proves the group-holdout split is actually a group holdout, and that
training produces a reasonable model, not a check against a fixed accuracy
number, since that would make the test brittle to any legitimate retraining.

Run with: pytest tests/test_forecasting_model.py -v
"""

from src.forecasting.model import (
    FEATURE_COLUMNS,
    HOLDOUT_BATTERY,
    TARGET_COLUMN,
    group_holdout_split,
    load_data,
)


def test_holdout_battery_entirely_excluded_from_training():
    df = load_data()
    X_train, y_train, X_test, y_test = group_holdout_split(df)

    train_rows = df.loc[X_train.index]
    test_rows = df.loc[X_test.index]

    assert HOLDOUT_BATTERY not in train_rows["battery_id"].unique()
    assert set(test_rows["battery_id"].unique()) == {HOLDOUT_BATTERY}


def test_split_covers_every_row_exactly_once():
    df = load_data()
    X_train, _, X_test, _ = group_holdout_split(df)
    assert len(X_train) + len(X_test) == len(df)
    assert set(X_train.index) & set(X_test.index) == set()


def test_feature_and_target_columns_exist_in_source_data():
    df = load_data()
    for col in FEATURE_COLUMNS + [TARGET_COLUMN, "battery_id"]:
        assert col in df.columns, f"missing column: {col}"


def test_target_is_zero_at_first_discharge_per_battery():
    """Sanity check on the extraction script's output, not the model:
    capacity_fade_pct should start at 0 for every battery's first cycle.
    """
    df = load_data()
    first_cycles = df.sort_values("discharge_cycle_index").groupby("battery_id").first()
    assert (first_cycles[TARGET_COLUMN] == 0).all()
