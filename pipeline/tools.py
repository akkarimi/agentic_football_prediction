import time
import requests
from ddgs import DDGS
from config import FOOTBALL_DATA_API_KEY

BASE_URL = "https://api.football-data.org/v4"
HEADERS  = {"X-Auth-Token": FOOTBALL_DATA_API_KEY}


def _get(endpoint: str, params: dict = None) -> dict:
    """GET request to Football-Data.org. Sleeps 6s to respect 10 req/min limit."""
    time.sleep(6)
    response = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, params=params or {})
    response.raise_for_status()
    return response.json()


def get_todays_matches(target_date: str) -> list[dict]:
    from datetime import date, timedelta
    next_day = (date.fromisoformat(target_date) + timedelta(days=1)).isoformat()
    data = _get("/matches", {"dateFrom": target_date, "dateTo": next_day})
    matches = data.get("matches", [])
    # Filter to only the requested date (UTC)
    return [m for m in matches if m.get("utcDate", "").startswith(target_date)]




def get_team_form(team_id: int, limit: int = 5) -> list[dict]:
    """Get last N finished matches for a team."""
    data = _get(f"/teams/{team_id}/matches", {"status": "FINISHED", "limit": limit})
    matches = data.get("matches", [])
    return sorted(matches, key=lambda m: m.get("utcDate", ""), reverse=True)[:limit]


def get_head_to_head(match_id: int, limit: int = 5) -> dict:
    """Get H2H record for a specific upcoming match."""
    return _get(f"/matches/{match_id}/head2head", {"limit": limit})


def get_standings(competition_code: str) -> dict:
    """Get current standings for a competition."""
    return _get(f"/competitions/{competition_code}/standings")


def search_news(query: str, max_results: int = 4) -> list[dict]:
    """Search for team news using DuckDuckGo (free, no API key needed)."""
    try:
        results = DDGS().text(query, max_results=max_results)
        return list(results) if results else []
    except Exception:
        return []

def get_match_result(match_id: int) -> dict | None:
    """Fetch actual score for a finished match."""
    try:
        data = _get(f"/matches/{match_id}")
        if data.get("status") == "FINISHED":
            score = data.get("score", {}).get("fullTime", {})
            hg = score.get("home")
            ag = score.get("away")
            if hg is not None and ag is not None:
                winner = "home" if hg > ag else ("away" if ag > hg else "draw")
                return {"home_goals": hg, "away_goals": ag, "winner": winner}
    except Exception as e:
        print(f"DEBUG result fetch error {match_id}: {e}")
    return None
