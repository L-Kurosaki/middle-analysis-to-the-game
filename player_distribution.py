"""
Player-level ball distribution analysis, linked to everything built so far.

Answers three things directly:
  1. DISPERSION: when a midfielder passes the ball, where does it go -
     to defenders, other midfielders, or forwards?
  2. COLLECTION: when a midfielder receives the ball, where did it come from?
  3. CONTRIBUTION SHARE: even though forwards score the goals, what share of a
     team's total ball involvement (passing + defensive actions combined) comes
     from each position group (GK / Defender / Midfielder / Forward) - split
     cleanly by Win / Draw / Loss, same structure as the earlier midfield analysis.

Data: StatsBomb Open Data, FIFA World Cup 2022 (competition_id=43, season_id=106)
"""
import pandas as pd
import numpy as np
import ast
from statsbombpy import sb
import warnings
warnings.filterwarnings("ignore")

POSITION_GROUP = {
    'Goalkeeper': 'Goalkeeper',
    'Right Center Back': 'Defender', 'Left Center Back': 'Defender', 'Center Back': 'Defender',
    'Right Back': 'Defender', 'Left Back': 'Defender',
    'Right Wing Back': 'Defender', 'Left Wing Back': 'Defender',
    'Center Defensive Midfield': 'Midfielder', 'Left Center Midfield': 'Midfielder',
    'Right Center Midfield': 'Midfielder', 'Center Attacking Midfield': 'Midfielder',
    'Right Defensive Midfield': 'Midfielder', 'Left Defensive Midfield': 'Midfielder',
    'Right Midfield': 'Midfielder', 'Left Midfield': 'Midfielder',
    'Right Attacking Midfield': 'Midfielder', 'Left Attacking Midfield': 'Midfielder',
    'Center Forward': 'Forward', 'Right Wing': 'Forward', 'Left Wing': 'Forward',
    'Right Center Forward': 'Forward', 'Left Center Forward': 'Forward',
}
DEF_EVENT_TYPES = ['Pressure', 'Interception', 'Block', 'Duel', 'Clearance', 'Ball Recovery']

def parse_loc(v):
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except Exception:
            return None
    return v if isinstance(v, list) else None

matches = pd.read_csv('data/matches.csv')
formations = pd.read_csv('data/formations.csv')
# lookup: (match_id, team, player) -> position group, for both passer AND recipient
formations['pos_group'] = formations['position'].map(POSITION_GROUP)
lookup = formations.set_index(['match_id', 'team', 'player'])['pos_group'].to_dict()

match_ids = matches['match_id'].tolist()

all_pass_flows = []   # for dispersion/collection matrices
all_contrib_rows = [] # for position-group contribution share by outcome
player_touch_rows = []  # for player-level average position / heatmap

for mid in match_ids:
    ev = sb.events(match_id=mid)
    m = matches[matches['match_id'] == mid].iloc[0]
    home, away = m['home_team'], m['away_team']
    home_score, away_score = m['home_score'], m['away_score']

    for team in [home, away]:
        opp = away if team == home else home
        gf, ga = (home_score, away_score) if team == home else (away_score, home_score)
        outcome = 'win' if gf > ga else ('loss' if gf < ga else 'draw')

        team_ev = ev[ev['team'] == team].copy()

        # ---------- Passes: dispersion (sender group -> recipient group) ----------
        passes = team_ev[team_ev['type'] == 'Pass'].copy()
        passes['loc'] = passes['location'].apply(parse_loc)
        passes['end_loc'] = passes['pass_end_location'].apply(parse_loc)
        passes['completed'] = passes['pass_outcome'].isna()
        passes['sender_group'] = passes['position'].map(POSITION_GROUP)
        passes['recipient_group'] = passes.apply(
            lambda r: lookup.get((mid, team, r['pass_recipient'])) if pd.notna(r.get('pass_recipient')) else None,
            axis=1
        )
        completed_passes = passes[passes['completed']]
        for _, r in completed_passes.dropna(subset=['sender_group', 'recipient_group']).iterrows():
            all_pass_flows.append({
                'match_id': mid, 'team': team, 'outcome': outcome,
                'sender_group': r['sender_group'], 'recipient_group': r['recipient_group']
            })

        # player-level average touch position (x = pitch length, higher = more attacking)
        passes_loc_valid = passes.dropna(subset=['loc'])
        for player, g in passes_loc_valid.groupby('player'):
            xs = g['loc'].apply(lambda l: l[0])
            player_touch_rows.append({
                'match_id': mid, 'team': team, 'player': player,
                'position_group': lookup.get((mid, team, player)),
                'avg_x': xs.mean(), 'n_passes': len(g)
            })

        # ---------- Contribution share: passes + defensive actions, by position group ----------
        def_ev = team_ev[team_ev['type'].isin(DEF_EVENT_TYPES)].copy()
        passes['group'] = passes['sender_group']
        def_ev['group'] = def_ev['position'].map(POSITION_GROUP)

        combined = pd.concat([
            passes[['group']].assign(action='pass'),
            def_ev[['group']].assign(action='defensive')
        ])
        total_actions = len(combined)
        group_counts = combined['group'].value_counts()
        for grp in ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']:
            share = (group_counts.get(grp, 0) / total_actions) if total_actions > 0 else np.nan
            all_contrib_rows.append({
                'match_id': mid, 'team': team, 'outcome': outcome,
                'position_group': grp, 'action_share': share, 'action_count': int(group_counts.get(grp, 0))
            })

pd.DataFrame(all_pass_flows).to_csv('data/pass_flows.csv', index=False)
pd.DataFrame(all_contrib_rows).to_csv('data/position_contribution.csv', index=False)
pd.DataFrame(player_touch_rows).to_csv('data/player_touch_position.csv', index=False)

print("Saved: pass_flows.csv, position_contribution.csv, player_touch_position.csv")
print(f"Pass flow rows: {len(all_pass_flows)} | Contribution rows: {len(all_contrib_rows)} | Player-touch rows: {len(player_touch_rows)}")
