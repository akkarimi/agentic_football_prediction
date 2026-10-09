SYSTEM_PROMPT = """You are an expert football (soccer) analyst. Your job is to analyze \
match data and produce accurate, data-driven predictions. Be objective. \
Weight recent form heavily. Factor in injuries and suspensions from the news. \
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

Based on all of the above, provide your prediction.

The three probabilities are fractions between 0 and 1 (e.g. 0.55, not 55) and must sum to 1.
home_win_prob is for {home_team}, away_win_prob is for {away_team}.
home_goals and away_goals are the single most likely final score, as whole numbers.
The prediction must agree with itself: predicted_winner must be the outcome with the highest
probability, and the score must show that same outcome (a draw means equal goals).

For the match_preview write 2-3 specific sentences covering: current form of both teams,
what could decide the match, and a tactical observation. Be specific about recent results.

Only name a player if that player is named in the LATEST NEWS & INJURIES section above.
Do not name any player from memory, in the preview or in key_factors. If the news names
no players, describe the teams without naming individuals.
"""
