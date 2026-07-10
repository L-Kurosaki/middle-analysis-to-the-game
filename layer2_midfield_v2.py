"""
Layer 2 v2: Midfielder contribution, properly separated into:
  (a) ON-BALL  - progressive passing contribution during attacking sequences
  (b) OFF-BALL - defensive positioning/actions when the team does NOT have the ball
Split cleanly into Win / Draw / Loss (not a binary).
Then combined with substitution behavior into one composite team-match score.

Data: StatsBomb open data, FIFA World Cup 2022 (competition_id=43, season_id=106)
Pitch coordinates: StatsBomb uses a 120 (x, direction of attack) x 80 (y) pitch.
"""
import pandas as pd
import numpy as np
import ast
from statsbombpy import sb
import warnings
warnings.filterwarnings("ignore")

MID_POSITIONS = [
    'Center Defensive Midfield', 'Left Center Midfield', 'Right Center Midfield',
    'Center Attacking Midfield', 'Right Defensive Midfield', 'Left Defensive Midfield',
    'Right Midfield', 'Left Midfield', 'Right Attacking Midfield', 'Left Attacking Midfield'
]
DEF_EVENT_TYPES = ['Pressure', 'Interception', 'Block', 'Duel', 'Clearance', 'Ball Recovery']

matches = pd.read_csv('data/matches.csv')
match_ids = matches['match_id'].tolist()

def parse_loc(v):
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except Exception:
            return None
    return v if isinstance(v, list) else None

rows = []

for mid in match_ids:
    ev = sb.events(match_id=mid)
    m = matches[matches['match_id'] == mid].iloc[0]
    home, away = m['home_team'], m['away_team']
    home_score, away_score = m['home_score'], m['away_score']

    for team in [home, away]:
        opp = away if team == home else home
        gf, ga = (home_score, away_score) if team == home else (away_score, home_score)
        outcome = 'win' if gf > ga else ('loss' if gf < ga else 'draw')

        team_ev = ev[ev['team'] == team]
        opp_ev = ev[ev['team'] == opp]

        # ---------- ON-BALL: progressive passing contribution by midfielders ----------
        passes = team_ev[team_ev['type'] == 'Pass'].copy()
        passes['loc'] = passes['location'].apply(parse_loc)
        passes['end_loc'] = passes['pass_end_location'].apply(parse_loc)
        passes = passes.dropna(subset=['loc', 'end_loc'])
        passes['start_x'] = passes['loc'].apply(lambda l: l[0])
        passes['end_x'] = passes['end_loc'].apply(lambda l: l[0])
        passes['progressive'] = (passes['end_x'] - passes['start_x']) >= 10  # moved ball forward 10+ meters
        passes['completed'] = passes['pass_outcome'].isna()
        passes['is_mid'] = passes['position'].isin(MID_POSITIONS)

        total_prog = passes.loc[passes['completed'], 'progressive'].sum()
        mid_prog = passes.loc[passes['completed'] & passes['is_mid'], 'progressive'].sum()
        mid_prog_share = (mid_prog / total_prog) if total_prog > 0 else np.nan

        # progressive passes that DIRECTLY precede a shot within the same possession (attacking contribution)
        shots = team_ev[team_ev['type'] == 'Shot']
        attacking_poss = shots['possession'].unique() if 'possession' in shots.columns else []
        mid_passes_in_attacks = passes[passes['is_mid'] & passes['possession'].isin(attacking_poss)] if 'possession' in passes.columns else pd.DataFrame()
        mid_attack_involvement = len(mid_passes_in_attacks)

        # ---------- OFF-BALL: defensive positioning/actions while NOT in possession ----------
        def_ev = team_ev[team_ev['type'].isin(DEF_EVENT_TYPES)].copy()
        def_ev['loc'] = def_ev['location'].apply(parse_loc)
        def_ev = def_ev.dropna(subset=['loc'])
        def_ev['x'] = def_ev['loc'].apply(lambda l: l[0])
        def_ev['is_mid'] = def_ev['position'].isin(MID_POSITIONS)

        mid_def_actions = def_ev[def_ev['is_mid']]
        # average x-position of midfield defensive actions -> how high up the pitch they defend
        # (StatsBomb x is 0-120, attacking direction; low x = deep/near own goal)
        mid_def_line_height = mid_def_actions['x'].mean() if len(mid_def_actions) > 0 else np.nan
        mid_def_action_count = len(mid_def_actions)
        total_def_actions = len(def_ev)
        mid_def_share = (mid_def_action_count / total_def_actions) if total_def_actions > 0 else np.nan

        rows.append({
            'match_id': mid, 'team': team, 'opponent': opp, 'outcome': outcome,
            'goals_for': gf, 'goals_against': ga,
            'mid_progressive_pass_share': mid_prog_share,
            'mid_attack_involvement': mid_attack_involvement,
            'mid_defensive_line_height': mid_def_line_height,
            'mid_defensive_action_share': mid_def_share,
            'mid_defensive_action_count': mid_def_action_count,
        })

df = pd.DataFrame(rows)
df.to_csv('data/layer2_v2_onball_offball.csv', index=False)

print("=" * 70)
print("ON-BALL vs OFF-BALL midfielder contribution, split Win / Draw / Loss")
print("=" * 70)
summary = df.groupby('outcome').agg(
    n=('team', 'count'),
    mid_progressive_share=('mid_progressive_pass_share', 'mean'),
    mid_attack_involvement=('mid_attack_involvement', 'mean'),
    mid_def_line_height=('mid_defensive_line_height', 'mean'),
    mid_def_action_share=('mid_defensive_action_share', 'mean'),
).round(3)
print(summary)
print()

# ---------- Combine with substitution behavior into one composite score ----------
subs = pd.read_csv('data/substitutions.csv')
sub_counts = subs.groupby(['match_id', 'team', 'score_state']).size().unstack(fill_value=0).reset_index()
for col in ['winning', 'losing', 'drawing']:
    if col not in sub_counts.columns:
        sub_counts[col] = 0

merged = df.merge(sub_counts, on=['match_id', 'team'], how='left').fillna(0)

# normalize each component 0-1 (percentile rank) so they combine fairly
def pctrank(s):
    return s.rank(pct=True)

merged['score_onball'] = pctrank(merged['mid_progressive_pass_share'].fillna(merged['mid_progressive_pass_share'].median()))
merged['score_offball'] = pctrank(merged['mid_defensive_action_share'].fillna(merged['mid_defensive_action_share'].median()))

# Resilience, done correctly: raw sub-count-while-losing is circular (you can only sub-while-losing
# if you were already losing, so it structurally rewards losing teams). Instead we use the actual
# GOAL-RATE LIFT achieved after those subs (from Layer 4) as the effectiveness signal, defaulting
# teams that never trailed to the tournament median (they had no chance to show this).
sub_effects = pd.read_csv('data/layer4_sub_effects.csv')
losing_effect = sub_effects[sub_effects['score_state'] == 'losing'].groupby(['match_id', 'team']).apply(
    lambda g: (g['goals_after_15min'] - g['goals_before_15min']).mean()
).rename('sub_effectiveness_while_losing').reset_index()

merged = merged.merge(losing_effect, on=['match_id', 'team'], how='left')
merged['sub_effectiveness_while_losing'] = merged['sub_effectiveness_while_losing'].fillna(
    merged['sub_effectiveness_while_losing'].median()
)
merged['score_resilience'] = pctrank(merged['sub_effectiveness_while_losing'])

merged['composite_midfield_index'] = (
    0.4 * merged['score_onball'] +
    0.35 * merged['score_offball'] +
    0.25 * merged['score_resilience']
) * 100

merged.to_csv('data/layer2_v2_composite.csv', index=False)

print("=" * 70)
print("COMPOSITE MIDFIELD INDEX (0-100) by outcome")
print("=" * 70)
comp_summary = merged.groupby('outcome')['composite_midfield_index'].agg(['mean', 'median', 'count']).round(1)
print(comp_summary)
print()
print("Top 10 teams by composite index:")
print(merged.sort_values('composite_midfield_index', ascending=False)[
    ['team', 'opponent', 'outcome', 'composite_midfield_index']
].head(10).to_string(index=False))
