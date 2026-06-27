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
- overtime (or stoppage_time/added_time/injury_time)
- stadium
- attendance
- temperature
- win/result/outcome/target (binary label; accepted values include `win`, `w`, `1`, `lose`, `loss`, `l`, `0`)

Upcoming CSV (`--upcoming-csv`) should include the same feature columns (without label).  
Optional: include `match_id` and `team` (or alias columns) so winners are selected per match.

### 3) Run training + prediction

```bash
python train_wc2026_xgboost.py \
  --train-csv ./data/wc_train.csv \
  --upcoming-csv ./data/wc2026_upcoming.csv \
  --output-csv ./data/predicted_winners_2026.csv \
  --model-json ./data/xgb_wc2026_model.json \
  --winner-threshold 0.5
```

The output CSV contains win probabilities and predicted winners (top probability per match if `match_id` is present).  
If multiple teams tie for max probability in the same match, all tied rows are marked `winner_tie`.
If `match_id` is absent, rows are labeled by `--winner-threshold` and only predicted winners are written.
