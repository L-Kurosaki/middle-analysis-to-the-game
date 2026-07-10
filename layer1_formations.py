"""Layer 1: Formation win rates"""
import pandas as pd

matches = pd.read_csv('data/matches.csv')
formations = pd.read_csv('data/formations.csv')

# one formation per team per match (starting XI formation)
team_formation = formations.groupby(['match_id','team'])['formation'].first().reset_index()

# attach result per team-perspective
rows = []
for _, m in matches.iterrows():
    for team, opp, gf, ga in [
        (m['home_team'], m['away_team'], m['home_score'], m['away_score']),
        (m['away_team'], m['home_team'], m['away_score'], m['home_score']),
    ]:
        outcome = 'win' if gf > ga else ('loss' if gf < ga else 'draw')
        rows.append({'match_id': m['match_id'], 'team': team, 'opponent': opp,
                      'goals_for': gf, 'goals_against': ga, 'outcome': outcome})
team_results = pd.DataFrame(rows)

merged = team_results.merge(team_formation, on=['match_id','team'], how='left')

summary = merged.groupby('formation').agg(
    matches=('outcome','count'),
    wins=('outcome', lambda x: (x=='win').sum()),
    draws=('outcome', lambda x: (x=='draw').sum()),
    losses=('outcome', lambda x: (x=='loss').sum()),
    avg_goals_for=('goals_for','mean'),
    avg_goals_against=('goals_against','mean'),
).reset_index()
summary['win_rate'] = (summary['wins'] / summary['matches']).round(3)
summary = summary[summary['matches'] >= 5].sort_values('win_rate', ascending=False)

summary.to_csv('data/layer1_formation_winrate.csv', index=False)
merged.to_csv('data/team_formation_results.csv', index=False)
print(summary.to_string(index=False))
