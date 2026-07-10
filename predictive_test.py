"""
Honest test: if you use these midfield metrics to predict match outcome,
how accurate is that really? Cross-validated, compared against baseline.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier

df = pd.read_csv('data/layer2_v2_composite.csv')
df = df.dropna(subset=['mid_progressive_pass_share','mid_defensive_action_share','composite_midfield_index'])

# Binary: win vs not-win (most common way this would actually get used - "will Team X win")
df['win_flag'] = (df['outcome']=='win').astype(int)

features = ['mid_progressive_pass_share','mid_defensive_action_share','composite_midfield_index']
X = df[features]
y = df['win_flag']

print(f"n = {len(df)}")
print(f"Base rate (always predict 'not win'): {(1-y.mean())*100:.1f}% accuracy just by guessing majority class")
print()

# Baseline: majority class
dummy = DummyClassifier(strategy='most_frequent')
dummy_scores = cross_val_score(dummy, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='accuracy')
print(f"Dummy (majority class) CV accuracy: {dummy_scores.mean()*100:.1f}%  (+/- {dummy_scores.std()*100:.1f})")

# Our model
model = make_pipeline(StandardScaler(), LogisticRegression())
scores = cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='accuracy')
print(f"Midfield-metrics model CV accuracy:  {scores.mean()*100:.1f}%  (+/- {scores.std()*100:.1f})")
print()

auc_scores = cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='roc_auc')
print(f"ROC-AUC (0.5 = coin flip, 1.0 = perfect): {auc_scores.mean():.3f}  (+/- {auc_scores.std():.3f})")
