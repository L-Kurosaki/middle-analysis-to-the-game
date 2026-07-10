"""Layer 4: Substitution behavior and effect - does subbing while losing actually help?"""
import pandas as pd
from statsbombpy import sb
import warnings
warnings.filterwarnings("ignore")

matches = pd.read_csv('data/matches.csv')
subs = pd.read_csv('data/substitutions.csv')

# 1. Sub COUNT by score state per team-match (behavior pattern)
sub_counts = subs.groupby(['match_id','team','score_state']).size().unstack(fill_value=0).reset_index()
print("Substitution counts by score-state at time of sub (sample):")
print(sub_counts.head(8).to_string())
print()

overall_avg = subs.groupby('score_state').size() / subs['match_id'].nunique()
print("Average subs per match by score-state at moment of substitution:")
print(subs['score_state'].value_counts())
print()

# 2. EFFECT: for teams losing at a substitution, do they score in the following 15 min more than base rate?
comp_id, season_id = 43, 106
match_ids = matches['match_id'].tolist()

results = []
for mid in match_ids:
    ev = sb.events(match_id=mid)
    m = matches[matches['match_id']==mid].iloc[0]
    home, away = m['home_team'], m['away_team']

    goal_events = ev[(ev['type']=='Shot') & (ev['shot_outcome']=='Goal')][['minute','second','team']].values.tolist()
    own_goals = ev[ev['type']=='Own Goal Against'][['minute','second','team']].values.tolist()
    goals = [(g[0], g[1], g[2]) for g in goal_events]
    for og in own_goals:
        scoring_team = away if og[2]==home else home
        goals.append((og[0], og[1], scoring_team))

    team_subs = subs[subs['match_id']==mid]
    for _, s in team_subs.iterrows():
        team = s['team']
        sub_min = s['minute']
        state = s['score_state']
        # goals scored BY this team in the 15 min after the sub
        goals_after = sum(1 for gm,gs,gt in goals if gt==team and sub_min < gm <= sub_min+15)
        goals_before = sum(1 for gm,gs,gt in goals if gt==team and max(0,sub_min-15) <= gm <= sub_min)
        results.append({'match_id': mid, 'team': team, 'minute': sub_min, 'score_state': state,
                         'goals_after_15min': goals_after, 'goals_before_15min': goals_before})

effect_df = pd.DataFrame(results)
effect_df.to_csv('data/layer4_sub_effects.csv', index=False)

print("Goal rate in 15min AFTER a substitution, by score-state at time of sub:")
print(effect_df.groupby('score_state')[['goals_after_15min','goals_before_15min']].mean().round(3))
print()
print("N substitutions per state:", effect_df['score_state'].value_counts().to_dict())
