SYSTEM_PROMPT = """You are an expert football (soccer) analyst. Your job is to analyze \
match data and produce accurate, data-driven predictions. Be objective. \
Weight recent form heavily. Factor in injuries and suspensions from the news. \
Do not speculate beyond what the data supports."""


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

Based on all of the above, provide your prediction including the expected number of goals each team will score (as decimals, e.g. 1.7).

For the match_preview write 2-3 specific sentences covering: current form of both teams, one or two key players who could decide the match, and a tactical observation. Be specific — mention actual player names and recent results."""
