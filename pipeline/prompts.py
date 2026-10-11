SYSTEM_PROMPT = """You are an expert football (soccer) analyst. Your job is to analyze \
match data and produce accurate, data-driven predictions. Be objective. \
Weight recent form heavily. Factor in injuries and suspensions from the news. \
Also weigh team morale: what the recent news says about each squad's confidence, unity and mood. \
Do not speculate beyond what the data supports. \
Your own knowledge of squads is out of date: players transfer and retire, so never state \
from memory which club a player plays for."""


def build_match_prompt(
    home_team: str,
    away_team: str,
    competition: str,
    match_date: str,
    home_form: str,
    away_form: str,
    h2h: str,
    standings: str,
    news: str,
) -> str:
    return f"""Analyze this upcoming match and predict the outcome.

MATCH: {home_team} vs {away_team}
COMPETITION: {competition}
DATE: {match_date}

HOME TEAM — {home_team} (last 5 matches):
{home_form}

AWAY TEAM — {away_team} (last 5 matches):
{away_form}

HEAD-TO-HEAD (last 5 meetings):
{h2h}

CURRENT STANDINGS:
{standings}

LATEST NEWS & INJURIES:
{news}

Before predicting, read the news for each team and judge its effect on the squad's psychology
and spirit: for example a manager under pressure or newly appointed, dressing-room conflict,
a morale-lifting or damaging recent result, fan unrest, contract or transfer sagas, off-field
controversy, or a returning key figure. Decide whether each team comes into the match lifted,
unsettled or unaffected, and let that shift the probabilities and score alongside form and injuries.
Base this only on the news above; if it says nothing relevant about a team's mood, treat morale
as neutral rather than guessing.

Based on all of the above, provide your prediction.

The three probabilities are fractions between 0 and 1 (e.g. 0.55, not 55) and must sum to 1.
home_win_prob is for {home_team}, away_win_prob is for {away_team}.
home_goals and away_goals are the single most likely final score, as whole numbers.
The prediction must agree with itself: predicted_winner must be the outcome with the highest
probability, and the score must show that same outcome (a draw means equal goals).

For the match_preview write 2-3 specific sentences covering: current form of both teams,
the mood and morale around each team according to the news, and what could decide the match.
Be specific about recent results. When morale influenced the prediction, include it in key_factors.

Only name a player if that player is named in the LATEST NEWS & INJURIES section above.
Do not name any player from memory, in the preview or in key_factors. If the news names
no players, describe the teams without naming individuals.
"""
