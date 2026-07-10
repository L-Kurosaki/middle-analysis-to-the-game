# World Cup 2022 Tactical Analysis — Source Code

Run in this order. Each script reads/writes CSVs in a local `data/` folder.

1. `build_dataset.py`
   Pulls all 64 matches of the FIFA World Cup 2022 from StatsBomb's open event data
   (via the `statsbombpy` package). Builds: matches.csv, formations.csv,
   substitutions.csv (tagged with score-state at the moment of each sub),
   passes.csv, shots.csv.

2. `layer1_formations.py`
   Formation win-rate analysis. Joins each team's starting formation to their
   match result, computes win/draw/loss rate and average goals for/against
   per formation (filtered to formations used 5+ times for a fair sample).

3. `layer2_midfield_v2.py`  ← the core "midfielder contribution" analysis
   Splits midfielder contribution into:
     - ON-BALL: share of progressive passes (moved the ball forward 10+ metres)
       that were made by central midfielders
     - OFF-BALL: share of the team's defensive actions (pressure, interception,
       block, duel, clearance, ball recovery) made by central midfielders while
       the team does NOT have the ball, plus average pitch-height (x-coordinate)
       of those actions
   Splits all of this cleanly into Win / Draw / Loss (not a binary).
   Combines on-ball + off-ball + substitution-effectiveness-while-losing into
   one 0–100 composite "midfield index" per team-match, using percentile
   normalization so the components combine fairly.
   NOTE on methodology: an earlier version of this composite used raw
   substitution COUNT while losing as a "resilience" factor. That's circular —
   a team can only sub while losing if it was already losing, so it structurally
   inflated losing teams' scores. This version replaces it with the actual
   GOAL-RATE LIFT achieved in the 15 minutes after those subs (pulled from
   layer4_subs.py's output), which measures effectiveness rather than just
   "were they behind."

4. `layer4_subs.py`
   Tags every substitution with the score-state (winning/losing/drawing) at
   that moment, then compares goals scored in the 15 minutes before vs. after
   each substitution, grouped by score-state. This is the effectiveness signal
   used by layer2_midfield_v2.py.

5. `layer3_contributions.py`
   Goal + assist leaderboard (goal contributions) as an impact proxy, since
   StatsBomb's open data doesn't include official Man-of-the-Match tags.

## Data source
StatsBomb Open Data, FIFA World Cup 2022 (competition_id=43, season_id=106),
pulled via the `statsbombpy` Python package. No API key required — this is
StatsBomb's free open dataset.

## Known limitations (say these out loud in the pitch)
- n=64 matches / 128 team-matches — one tournament is underpowered for strong
  statistical claims. The on-ball/off-ball gaps by outcome are directionally
  consistent with the hypothesis but small.
- "Off-ball positioning" is approximated from defensive EVENT locations
  (pressures, tackles, etc.), not full player tracking — StatsBomb's open data
  doesn't include tracking/freeze-frame data for most matches, so this is a
  reasonable proxy, not true positional tracking.
- The composite index's weights (0.4 / 0.35 / 0.25) are a judgment call, not
  fitted — worth saying that plainly if asked.

## Predictive accuracy test (added after initial build)

6. `predictive_test.py`
   Quick sanity check: does the World Cup composite index predict win/not-win using
   simple 5-fold cross-validation? Answer: no (AUC 0.503, a coin flip) - this is what
   prompted the more rigorous test below.

7. `build_predictive_dataset.py`
   Pulls a FULL LEAGUE SEASON (La Liga 2015/16, competition_id=11, season_id=27,
   380 matches, 20 teams) rather than a knockout tournament, because genuine
   prediction needs enough prior matches per team to build rolling form features.
   Checkpoints every 20 matches to data/laliga_team_match_features.csv so a network
   hiccup doesn't lose progress - safe to re-run, it resumes from the checkpoint.

8. `predictive_model_v2.py`
   The actual predictive test, done properly:
     - Rolling averages use ONLY each team's prior 5 matches (shift(1) before rolling)
       - never the match being predicted. This avoids the single most common mistake
       in sports prediction projects: leaking the outcome you're trying to predict
       into the features that predict it.
     - TIME-ORDERED train/test split (first 70% of season -> train, last 30% -> test),
       not random shuffling, since shuffling would let mid-season data "predict"
       early-season matches, which can't happen in reality.
     - Compares three feature sets against a naive "home team always wins" baseline.

   RESULT: midfield metrics alone (47.2% accuracy, AUC 0.476) perform WORSE than the
   naive baseline (50.9%, AUC 0.500). Recent form/points alone (58.5%, AUC 0.609) is
   a real, legitimate predictor. Combining both makes it worse (52.8%) - the midfield
   metrics dilute a good feature rather than adding to it.

   TAKEAWAY: the on-ball/off-ball midfield metrics are a valid IN-MATCH DESCRIPTIVE
   tool (what's happening in front of you right now) but not a PRE-MATCH PREDICTOR.
   Don't claim they predict results - the data doesn't support that claim, and this
   test is the receipt if anyone asks.

## Player-level ball distribution (added after predictive test)

9. `player_distribution.py`
   Links every completed pass to BOTH the sender's and receiver's position group
   (Goalkeeper / Defender / Midfielder / Forward), using formations.csv as the
   player -> starting position lookup. Produces three files:
     - `pass_flows.csv` - every pass tagged sender_group -> recipient_group,
       used to compute midfielder DISPERSION (where their passes go) and
       COLLECTION (where their received passes come from).
     - `position_contribution.csv` - for each team-match, what share of TOTAL
       ball actions (passes + defensive actions combined) came from each
       position group, split Win / Draw / Loss.
     - `player_touch_position.csv` - per-player average pitch position (x-coord)
       and pass volume per match, for individual-player pattern checks.

   KEY RESULTS:
     - Midfielders send 51.7% of their passes BACKWARD to defenders and only
       18.9% FORWARD to forwards - the "link" role is real but asymmetric,
       mostly recycling with the back line rather than launching attacks directly.
       Same pattern on collection: 52.9% received from defenders, 16.3% from
       forwards.
     - Defenders always carry the largest raw share of team ball actions
       (42-46%), regardless of result - a function of build-up starting deep,
       not an importance signal.
     - The share that shifts MOST with winning is forwards (16.4% in losses ->
       18.5% in wins, +13% relative), not midfielders (32.9% -> 34.4%, a smaller
       move in the same direction). Direct answer to "who contributes most to
       winning despite forwards scoring the goals": forward involvement itself
       rises the most in wins, not just their goal output.

## Known limitation on this section
Recipient position is looked up from STARTING lineup position (formations.csv),
not their position at the exact moment of the pass, and substitutes not in the
starting XI have no lookup entry, so some pass-flow rows are dropped (~2% of
completed passes had no resolvable recipient group). This is a reasonable
approximation, not perfect ground truth - worth saying if asked.
