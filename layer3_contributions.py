"""Layer 3: Goal contributions (goals + assists) per player as impact proxy (no official MOTM in open data)"""
import pandas as pd

shots = pd.read_csv('data/shots.csv')

goals = shots[shots['shot_outcome']=='Goal'].groupby('player').size().rename('goals')
# assists = key passes leading to a goal (shot_key_pass_id present on the goal-scoring shot -> attribute to passer)
# we need pass id -> player mapping
passes = pd.read_csv('data/passes.csv')

# shot_key_pass_id links to a pass event id; we don't have pass event id in passes.csv currently, so approximate via assists using statsbomb 'pass_goal_assist' flag instead
from statsbombpy import sb
import warnings
warnings.filterwarnings("ignore")
matches = pd.read_csv('data/matches.csv')

assist_counts = {}
for mid in matches['match_id']:
    ev = sb.events(match_id=mid)
    a = ev[(ev['type']=='Pass') & (ev.get('pass_goal_assist')==True)]
    for p in a['player'].dropna():
        assist_counts[p] = assist_counts.get(p,0)+1

assists = pd.Series(assist_counts, name='assists')
contrib = pd.concat([goals, assists], axis=1).fillna(0)
contrib['goal_contributions'] = contrib['goals'] + contrib['assists']
contrib = contrib.sort_values('goal_contributions', ascending=False)
contrib.to_csv('data/layer3_goal_contributions.csv')
print(contrib.head(15).to_string())
