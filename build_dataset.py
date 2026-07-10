"""
Build core match-level dataset from StatsBomb World Cup 2022 open data.
Captures: formation, result, substitution timing/score-state, midfielder pass involvement,
goal contributions (goals+assists) per player.
"""
import pandas as pd
from statsbombpy import sb
import warnings
warnings.filterwarnings("ignore")

comp_id, season_id = 43, 106  # FIFA World Cup 2022

matches = sb.matches(competition_id=comp_id, season_id=season_id)
match_ids = matches['match_id'].tolist()
print(f"Total matches: {len(match_ids)}")

all_subs = []
all_passes = []
all_shots = []
all_formations = []
all_match_meta = []

for i, mid in enumerate(match_ids):
    try:
        ev = sb.events(match_id=mid)
    except Exception as e:
        print(f"Failed match {mid}: {e}")
        continue

    row = matches[matches['match_id'] == mid].iloc[0]
    home, away = row['home_team'], row['away_team']
    home_score, away_score = row['home_score'], row['away_score']

    # --- Formations (Starting XI + Tactical Shifts) ---
    startxi = ev[ev['type'] == 'Starting XI']
    for _, r in startxi.iterrows():
        tactics = r['tactics']
        if isinstance(tactics, dict):
            formation = tactics.get('formation')
            for p in tactics.get('lineup', []):
                all_formations.append({
                    'match_id': mid, 'team': r['team'], 'formation': formation,
                    'player': p['player']['name'], 'position': p['position']['name']
                })

    # --- Substitutions with minute + running score state ---
    ev_sorted = ev.sort_values(['minute', 'second'])
    home_g, away_g = 0, 0
    goal_events = ev[(ev['type'] == 'Shot') & (ev['shot_outcome'] == 'Goal')]
    own_goals = ev[ev['type'] == 'Own Goal Against']

    # build minute-by-minute score timeline
    score_timeline = []  # (minute, home_g, away_g)
    goals_list = []
    for _, g in goal_events.iterrows():
        goals_list.append((g['minute'], g['second'], g['team']))
    for _, og in own_goals.iterrows():
        scoring_team = away if og['team'] == home else home
        goals_list.append((og['minute'], og['second'], scoring_team))
    goals_list.sort(key=lambda x: (x[0], x[1]))

    def score_at(minute, second):
        h, a = 0, 0
        for gm, gs, gteam in goals_list:
            if (gm, gs) <= (minute, second):
                if gteam == home:
                    h += 1
                else:
                    a += 1
        return h, a

    subs = ev[ev['type'] == 'Substitution']
    for _, s in subs.iterrows():
        h_g, a_g = score_at(s['minute'], s['second'])
        team = s['team']
        if team == home:
            diff = h_g - a_g
        else:
            diff = a_g - h_g
        state = 'winning' if diff > 0 else ('losing' if diff < 0 else 'drawing')
        all_subs.append({
            'match_id': mid, 'team': team, 'minute': s['minute'],
            'player_off': s.get('player'), 'player_on': s.get('substitution_replacement'),
            'score_state': state, 'score_diff': diff
        })

    # --- Passes (for midfielder involvement / pass network) ---
    passes = ev[ev['type'] == 'Pass'][['match_id','team','player','position','pass_outcome','pass_recipient']]
    all_passes.append(passes)

    # --- Goal contributions (goals + assists) ---
    shots = ev[ev['type'] == 'Shot'][['match_id','team','player','shot_outcome','shot_key_pass_id']]
    all_shots.append(shots)

    all_match_meta.append({
        'match_id': mid, 'home_team': home, 'away_team': away,
        'home_score': home_score, 'away_score': away_score,
        'result': 'home_win' if home_score > away_score else ('away_win' if away_score > home_score else 'draw')
    })

    if (i+1) % 10 == 0:
        print(f"Processed {i+1}/{len(match_ids)} matches")

pd.DataFrame(all_formations).to_csv('data/formations.csv', index=False)
pd.DataFrame(all_subs).to_csv('data/substitutions.csv', index=False)
pd.concat(all_passes).to_csv('data/passes.csv', index=False)
pd.concat(all_shots).to_csv('data/shots.csv', index=False)
pd.DataFrame(all_match_meta).to_csv('data/matches.csv', index=False)

print("Done. Files saved to data/")
