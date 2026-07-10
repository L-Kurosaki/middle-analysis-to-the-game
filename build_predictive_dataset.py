"""
Predictive dataset build: La Liga 2015/16, full season (380 matches, 20 teams).
Extracts per-team-match: progressive pass share (midfield), defensive action share (midfield),
plus goals/result, for later use as ROLLING FORM features (computed from each team's
PRIOR matches only - no leakage - to actually predict future matches).
Checkpoints to disk every 20 matches so a network hiccup doesn't lose all progress.
"""
import pandas as pd
import numpy as np
import ast
import os
from statsbombpy import sb
import warnings
warnings.filterwarnings("ignore")

comp_id, season_id = 11, 27
CKPT = 'data/laliga_team_match_features.csv'

MID_POSITIONS = [
    'Center Defensive Midfield', 'Left Center Midfield', 'Right Center Midfield',
    'Center Attacking Midfield', 'Right Defensive Midfield', 'Left Defensive Midfield',
    'Right Midfield', 'Left Midfield', 'Right Attacking Midfield', 'Left Attacking Midfield'
]
DEF_EVENT_TYPES = ['Pressure', 'Interception', 'Block', 'Duel', 'Clearance', 'Ball Recovery']

def parse_loc(v):
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except Exception:
            return None
    return v if isinstance(v, list) else None

matches = sb.matches(competition_id=comp_id, season_id=season_id)
matches = matches.sort_values('match_date').reset_index(drop=True)
match_ids = matches['match_id'].tolist()

done_ids = set()
if os.path.exists(CKPT):
    prev = pd.read_csv(CKPT)
    done_ids = set(prev['match_id'].unique())
    print(f"Resuming: {len(done_ids)} matches already done")
else:
    prev = pd.DataFrame()

rows = []
for i, mid in enumerate(match_ids):
    if mid in done_ids:
        continue
    try:
        ev = sb.events(match_id=mid)
    except Exception as e:
        print(f"FAILED match {mid}: {e}")
        continue

    m = matches[matches['match_id'] == mid].iloc[0]
    home, away = m['home_team'], m['away_team']
    home_score, away_score = m['home_score'], m['away_score']
    match_date = m['match_date']

    for team in [home, away]:
        opp = away if team == home else home
        gf, ga = (home_score, away_score) if team == home else (away_score, home_score)
        outcome = 'win' if gf > ga else ('loss' if gf < ga else 'draw')
        is_home = (team == home)

        team_ev = ev[ev['team'] == team]

        passes = team_ev[team_ev['type'] == 'Pass'].copy()
        passes['loc'] = passes['location'].apply(parse_loc)
        passes['end_loc'] = passes['pass_end_location'].apply(parse_loc)
        passes = passes.dropna(subset=['loc', 'end_loc'])
        if len(passes) > 0:
            passes['start_x'] = passes['loc'].apply(lambda l: l[0])
            passes['end_x'] = passes['end_loc'].apply(lambda l: l[0])
            passes['progressive'] = (passes['end_x'] - passes['start_x']) >= 10
            passes['completed'] = passes['pass_outcome'].isna()
            passes['is_mid'] = passes['position'].isin(MID_POSITIONS)
            total_prog = passes.loc[passes['completed'], 'progressive'].sum()
            mid_prog = passes.loc[passes['completed'] & passes['is_mid'], 'progressive'].sum()
            mid_prog_share = (mid_prog / total_prog) if total_prog > 0 else np.nan
        else:
            mid_prog_share = np.nan

        def_ev = team_ev[team_ev['type'].isin(DEF_EVENT_TYPES)].copy()
        if len(def_ev) > 0:
            def_ev['is_mid'] = def_ev['position'].isin(MID_POSITIONS)
            mid_def_share = def_ev['is_mid'].mean()
        else:
            mid_def_share = np.nan

        rows.append({
            'match_id': mid, 'match_date': match_date, 'team': team, 'opponent': opp,
            'is_home': is_home, 'goals_for': gf, 'goals_against': ga, 'outcome': outcome,
            'mid_progressive_pass_share': mid_prog_share,
            'mid_defensive_action_share': mid_def_share,
        })

    if len(rows) >= 20:
        pd.concat([prev, pd.DataFrame(rows)]).to_csv(CKPT, index=False)
        prev = pd.read_csv(CKPT)
        rows = []
        print(f"Checkpoint saved: {i+1}/{len(match_ids)} matches processed")

# final flush
if rows:
    pd.concat([prev, pd.DataFrame(rows)]).to_csv(CKPT, index=False)

final = pd.read_csv(CKPT)
print(f"\nDONE. Total team-match rows: {len(final)} across {final['match_id'].nunique()} matches")
