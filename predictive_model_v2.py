"""
Genuine predictive test: for each match, use each team's ROLLING AVERAGE from their
PRIOR N matches only (never the current or future match - that would be leakage)
as pre-match features, then predict the outcome.

Compared against:
  - Majority-class baseline
  - Home-advantage-only baseline (always predict home win) - the real-world naive baseline
Uses TIME-ORDERED cross-validation (expanding window), not random shuffling, since
shuffling randomly would let future matches leak into "past" rolling averages.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

df = pd.read_csv('data/laliga_team_match_features.csv')
df['match_date'] = pd.to_datetime(df['match_date'])
df = df.sort_values(['team', 'match_date']).reset_index(drop=True)

ROLL_N = 5  # form window: last 5 matches

# rolling average of EACH TEAM's own metrics from their prior N matches (shift(1) excludes current match)
for col in ['mid_progressive_pass_share', 'mid_defensive_action_share']:
    df[f'{col}_roll'] = (
        df.groupby('team')[col]
        .transform(lambda s: s.shift(1).rolling(ROLL_N, min_periods=3).mean())
    )

# rolling win rate (recent form) - a standard, simple predictive feature to compare against
df['win_flag'] = (df['outcome'] == 'win').astype(int)
df['points'] = df['outcome'].map({'win': 3, 'draw': 1, 'loss': 0})
df['recent_points_roll'] = (
    df.groupby('team')['points']
    .transform(lambda s: s.shift(1).rolling(ROLL_N, min_periods=3).mean())
)

# now build ONE ROW PER MATCH: home team's rolling features vs away team's rolling features
home_rows = df[df['is_home'] == True].copy()
away_rows = df[df['is_home'] == False].copy()

merged = home_rows.merge(
    away_rows, on='match_id', suffixes=('_home', '_away')
)

merged['target_home_win'] = (merged['outcome_home'] == 'win').astype(int)
merged = merged.dropna(subset=[
    'mid_progressive_pass_share_roll_home', 'mid_defensive_action_share_roll_home',
    'mid_progressive_pass_share_roll_away', 'mid_defensive_action_share_roll_away',
    'recent_points_roll_home', 'recent_points_roll_away'
])
merged = merged.sort_values('match_date_home').reset_index(drop=True)

print(f"Matches usable after rolling window warm-up: {len(merged)} / 380")
print(f"(early-season matches dropped - not enough prior history yet for a fair test)\n")

feature_sets = {
    'Midfield metrics only': ['mid_progressive_pass_share_roll_home', 'mid_defensive_action_share_roll_home',
                                'mid_progressive_pass_share_roll_away', 'mid_defensive_action_share_roll_away'],
    'Recent form (points) only': ['recent_points_roll_home', 'recent_points_roll_away'],
    'Midfield metrics + form': ['mid_progressive_pass_share_roll_home', 'mid_defensive_action_share_roll_home',
                                  'mid_progressive_pass_share_roll_away', 'mid_defensive_action_share_roll_away',
                                  'recent_points_roll_home', 'recent_points_roll_away'],
}

y = merged['target_home_win']
n = len(merged)
split = int(n * 0.7)  # time-ordered: train on first 70% of season, test on final 30%

print(f"Train: first {split} matches (early-mid season) | Test: last {n-split} matches (late season)\n")
print(f"Base rate - home team wins: {y.mean()*100:.1f}% of matches")
print(f"Baseline (always predict 'home wins'): {y.iloc[split:].mean()*100:.1f}% accuracy on test set\n")

results = {}
for name, feats in feature_sets.items():
    X = merged[feats]
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    results[name] = (acc, auc)
    print(f"{name:30s} -> Accuracy: {acc*100:5.1f}%   AUC: {auc:.3f}")

print()
always_home = accuracy_score(y_test, np.ones(len(y_test)))
print(f"{'Always predict home win':30s} -> Accuracy: {always_home*100:5.1f}%   AUC: 0.500 (by definition)")
