# xgb-wc2026
XGBoost simulation using World Cup match statistics

## Train and predict 2026 winners

Use the script below to train a binary win/lose XGBoost model and predict likely winners for upcoming 2026 matches.

### 1) Install dependencies

```bash
pip install pandas numpy xgboost
```

### 2) Prepare input CSV files

Training CSV (`--train-csv`) should include:

- goals
- penalties
- max_goal_ratio (or equivalent alias)
- ball_control_pct (or equivalent alias)
- blocked_goals
- corner_kicks
- overtime (or overage_time)
- stadium
- attendance
- temperature
- win/result/outcome/target (binary win/lose label)

Upcoming CSV (`--upcoming-csv`) should include the same feature columns (without label).  
Optional: include `match_id` and `team` (or alias columns) so winners are selected per match.

### 3) Run training + prediction

```bash
python train_wc2026_xgboost.py \
  --train-csv /absolute/path/to/wc_train.csv \
  --upcoming-csv /absolute/path/to/wc2026_upcoming.csv \
  --output-csv /absolute/path/to/predicted_winners_2026.csv \
  --model-json /absolute/path/to/xgb_wc2026_model.json
```

The output CSV contains win probabilities and predicted winners (top probability per match if `match_id` is present).
