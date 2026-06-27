#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


FEATURE_ALIASES = {
    "goals": ["goals", "goal"],
    "penalties": ["penalties", "penalty"],
    "max_goal_ratio": [
        "max_goal_ratio",
        "max",
        "max_goals_by_player_total_goals",
        "max_goals_by_player_over_total_goals",
        "max_goals_ratio",
    ],
    "ball_control_pct": [
        "ball_control_pct",
        "ball_control_percentage",
        "ball_control",
        "possession",
        "possession_pct",
    ],
    "blocked_goals": ["blocked_goals", "blocked_goal", "blocks"],
    "corner_kicks": ["corner_kicks", "corners", "corner"],
    "overtime": ["overtime", "overage_time", "stoppage_time", "extra_time"],
    "stadium": ["stadium", "venue"],
    "attendance": ["attendance", "crowd"],
    "temperature": ["temperature", "temp", "weather_temperature"],
}

LABEL_ALIASES = ["win", "result", "outcome", "target"]
MATCH_ID_ALIASES = ["match_id", "fixture_id", "game_id", "match"]
TEAM_ALIASES = ["team", "team_name", "country"]


def normalize_name(name: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", name.strip().lower())).strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_name(c) for c in out.columns]
    return out


def pick_column(df: pd.DataFrame, aliases: list[str], required: bool = True) -> str | None:
    for alias in aliases:
        if alias in df.columns:
            return alias
    if required:
        raise ValueError(f"Missing required column. Tried aliases: {aliases}")
    return None


def find_feature_columns(df: pd.DataFrame) -> dict[str, str]:
    feature_cols = {}
    for canonical, aliases in FEATURE_ALIASES.items():
        feature_cols[canonical] = pick_column(df, aliases, required=True)
    return feature_cols


def parse_percent(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value = value.strip().replace("%", "")
    return pd.to_numeric(value, errors="coerce")


def to_numeric_series(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def build_features(df: pd.DataFrame, feature_map: dict[str, str]) -> pd.DataFrame:
    numeric = pd.DataFrame(
        {
            "goals": to_numeric_series(df[feature_map["goals"]]),
            "penalties": to_numeric_series(df[feature_map["penalties"]]),
            "max_goal_ratio": to_numeric_series(df[feature_map["max_goal_ratio"]]),
            "ball_control_pct": df[feature_map["ball_control_pct"]].map(parse_percent),
            "blocked_goals": to_numeric_series(df[feature_map["blocked_goals"]]),
            "corner_kicks": to_numeric_series(df[feature_map["corner_kicks"]]),
            "overtime": to_numeric_series(df[feature_map["overtime"]]),
            "attendance": to_numeric_series(df[feature_map["attendance"]]),
            "temperature": to_numeric_series(df[feature_map["temperature"]]),
        }
    )
    stadium = pd.get_dummies(df[feature_map["stadium"]].fillna("unknown").astype(str), prefix="stadium")
    merged = pd.concat([numeric, stadium], axis=1)
    return merged.fillna(merged.median(numeric_only=True)).fillna(0)


def parse_label(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        lowered = series.astype(str).str.strip().str.lower()
        return lowered.map(
            {
                "win": 1,
                "w": 1,
                "1": 1,
                "lose": 0,
                "loss": 0,
                "l": 0,
                "0": 0,
            }
        )
    return pd.to_numeric(series, errors="coerce")


def train_model(x: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=350,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(x, y)
    return model


def add_missing_columns(pred_x: pd.DataFrame, train_columns: pd.Index) -> pd.DataFrame:
    out = pred_x.copy()
    for col in train_columns:
        if col not in out.columns:
            out[col] = 0
    return out[train_columns]


def main():
    parser = argparse.ArgumentParser(
        description="Train an XGBoost model on World Cup stats and predict likely winners."
    )
    parser.add_argument("--train-csv", required=True, help="CSV with historical matches and win/lose labels.")
    parser.add_argument("--upcoming-csv", required=True, help="CSV with upcoming 2026 match/team stats.")
    parser.add_argument("--output-csv", default="predicted_winners_2026.csv", help="Output CSV for predictions.")
    parser.add_argument("--model-json", default="xgb_wc2026_model.json", help="Path to save trained model JSON.")
    args = parser.parse_args()

    train_df = normalize_columns(pd.read_csv(args.train_csv))
    upcoming_df = normalize_columns(pd.read_csv(args.upcoming_csv))

    label_col = pick_column(train_df, LABEL_ALIASES, required=True)
    feature_map = find_feature_columns(train_df)

    x_train = build_features(train_df, feature_map)
    y_train = parse_label(train_df[label_col]).dropna()
    x_train = x_train.loc[y_train.index]
    y_train = y_train.astype(int)

    if y_train.nunique() < 2:
        raise ValueError("Training label has fewer than 2 classes after parsing win/lose values.")

    model = train_model(x_train, y_train)
    model.save_model(args.model_json)

    upcoming_feature_map = find_feature_columns(upcoming_df)
    x_upcoming = build_features(upcoming_df, upcoming_feature_map)
    x_upcoming = add_missing_columns(x_upcoming, x_train.columns)
    win_probability = model.predict_proba(x_upcoming)[:, 1]

    results = upcoming_df.copy()
    results["win_probability"] = win_probability

    match_col = pick_column(results, MATCH_ID_ALIASES, required=False)
    team_col = pick_column(results, TEAM_ALIASES, required=False)

    if match_col:
        winner_rows = results.loc[results.groupby(match_col)["win_probability"].idxmax()].copy()
        winner_rows["predicted_result"] = "winner"
    else:
        winner_rows = results.sort_values("win_probability", ascending=False).copy()
        winner_rows["predicted_result"] = np.where(winner_rows["win_probability"] >= 0.5, "winner", "loser")

    preferred_columns = [c for c in [match_col, team_col, "win_probability", "predicted_result"] if c]
    winner_rows = winner_rows[preferred_columns + [c for c in winner_rows.columns if c not in preferred_columns]]

    output_path = Path(args.output_csv)
    winner_rows.to_csv(output_path, index=False)

    print(f"Saved model to: {Path(args.model_json).resolve()}")
    print(f"Saved predictions to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
