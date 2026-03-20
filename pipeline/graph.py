from typing import TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import Literal

from pipeline.tools import get_match_result
from db import get_unresolved_predictions, update_actual_result


from config import GROQ_API_KEY, GROQ_MODEL
from pipeline.tools import (
    get_todays_matches, get_team_form, get_head_to_head,
    get_standings, search_news
)
from pipeline.prompts import SYSTEM_PROMPT, build_match_prompt


# --- Output schema ---

class Prediction(BaseModel):
    home_win_prob:    float
    draw_prob:        float
    away_win_prob:    float
    home_goals:       float
    away_goals:       float
    confidence:       Literal["low", "medium", "high"]
    key_factors:      list[str]
    match_preview: str = Field(description="A detailed 2-3 sentence match preview covering recent form, key players to watch, and tactical outlook. Must be specific to these two teams.")
    predicted_winner: Literal["home", "draw", "away"]



# --- State ---

class State(TypedDict):
    target_date:     str
    matches:         list[dict]
    enriched:        list[dict]
    predictions:     list[dict]


# --- Formatting helpers ---

def _fmt_form(matches: list[dict]) -> str:
    if not matches:
        return "No data available"
    lines = []
    for m in matches:
        home  = m.get("homeTeam", {}).get("name", "?")
        away  = m.get("awayTeam", {}).get("name", "?")
        score = m.get("score", {}).get("fullTime", {})
        hg    = score.get("home", "?")
        ag    = score.get("away", "?")
        date  = m.get("utcDate", "")[:10]
        lines.append(f"{date}: {home} {hg}–{ag} {away}")
    return "\n".join(lines)


def _fmt_h2h(data: dict) -> str:
    matches = data.get("matches", [])
    if not matches:
        return "No H2H data available"
    agg       = data.get("aggregates", {})
    home_wins = agg.get("homeTeam", {}).get("wins", "?")
    away_wins = agg.get("awayTeam", {}).get("wins", "?")
    draws     = agg.get("homeTeam", {}).get("draws", "?")
    lines     = [f"Record: Home {home_wins}W / {draws}D / {away_wins}W Away"]
    for m in matches[:3]:
        home  = m.get("homeTeam", {}).get("name", "?")
        away  = m.get("awayTeam", {}).get("name", "?")
        score = m.get("score", {}).get("fullTime", {})
        date  = m.get("utcDate", "")[:10]
        lines.append(f"{date}: {home} {score.get('home','?')}–{score.get('away','?')} {away}")
    return "\n".join(lines)


def _fmt_standings(data: dict, home: str, away: str) -> str:
    if not data:
        return "Not available"
    table = (data.get("standings") or [{}])[0].get("table", [])
    lines = []
    for e in table:
        name = e.get("team", {}).get("name", "")
        if home in name or away in name:
            lines.append(
                f"{e.get('position')}. {name} — "
                f"P{e.get('playedGames')} "
                f"Pts:{e.get('points')} "
                f"GD:{e.get('goalDifference')}"
            )
    return "\n".join(lines) if lines else "Not available"


def _fmt_news(items: list[dict]) -> str:
    if not items:
        return "No recent news found"
    lines = []
    for item in items[:6]:
        title = item.get("title", "")
        body  = item.get("body", item.get("snippet", ""))[:150]
        if title:
            lines.append(f"• {title}: {body}")
    return "\n".join(lines)


# --- Nodes ---

TRACKED_TEAMS = {
    "manchester united", "real madrid", "arsenal",
    "chelsea", "manchester city", "bayern", "dortmund",
    "inter milan", "ac milan", "paris saint-germain", "psg", 
    "fc barcelona", "liverpool", "atletico madrid", "juventus",
    "napoli"
}

def _is_tracked(match: dict) -> bool:
    home = match.get("homeTeam", {}).get("name", "").lower()
    away = match.get("awayTeam", {}).get("name", "").lower()
    return any(t in home or t in away for t in TRACKED_TEAMS)


def fetch_fixtures(state: State) -> State:
    try:
        matches = get_todays_matches(state["target_date"])
        matches = [m for m in matches if _is_tracked(m)]
    except Exception as e:
        print("DEBUG fetch_fixtures ERROR:", e)
        matches = []
    return {**state, "matches": matches}




def enrich_matches(state: State) -> State:
    enriched = []
    for match in state["matches"]:
        try:
            home_id   = match["homeTeam"]["id"]
            away_id   = match["awayTeam"]["id"]
            match_id  = match["id"]
            comp_code = match["competition"]["code"]
            home_name = match["homeTeam"]["name"]
            away_name = match["awayTeam"]["name"]

            home_form = get_team_form(home_id)
            away_form = get_team_form(away_id)
            h2h       = get_head_to_head(match_id)

            try:
                standings = get_standings(comp_code)
            except Exception:
                standings = {}

            home_news = search_news(f"{home_name} football injury team news 2025")
            away_news = search_news(f"{away_name} football injury team news 2025")

            enriched.append({
                **match,
                "home_form": home_form,
                "away_form": away_form,
                "h2h":       h2h,
                "standings": standings,
                "news":      home_news + away_news,
            })
        except Exception as e:
            enriched.append({**match, "enrich_error": str(e)})

    return {**state, "enriched": enriched}


def predict_matches(state: State) -> State:
    llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.1)
    structured_llm = llm.with_structured_output(Prediction)

    predictions = []
    for match in state["enriched"]:
        home_name = match["homeTeam"]["name"]
        away_name = match["awayTeam"]["name"]

        if "enrich_error" in match:
            predictions.append({
                "match_id":  match["id"],
                "home_team": home_name,
                "away_team": away_name,
                "competition": match["competition"]["name"],
                "match_date":  match["utcDate"][:10],
                "error": match["enrich_error"],
                "home_goals": result.home_goals,
                "away_goals": result.away_goals,

            })
            continue

        prompt = build_match_prompt(
            home_team   = home_name,
            away_team   = away_name,
            competition = match["competition"]["name"],
            match_date  = match["utcDate"][:10],
            home_form   = _fmt_form(match.get("home_form", [])),
            away_form   = _fmt_form(match.get("away_form", [])),
            h2h         = _fmt_h2h(match.get("h2h", {})),
            standings   = _fmt_standings(match.get("standings", {}), home_name, away_name),
            news        = _fmt_news(match.get("news", [])),
        )

        try:
            result = structured_llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            predictions.append({
                "match_id":       match["id"],
                "home_team":      home_name,
                "away_team":      away_name,
                "competition":    match["competition"]["name"],
                "match_date":     match["utcDate"][:10],
                "utc_date":       match["utcDate"],
                "home_win_prob":  result.home_win_prob,
                "draw_prob":      result.draw_prob,
                "away_win_prob":  result.away_win_prob,
                "home_goals":     result.home_goals,
                "away_goals":     result.away_goals,
                "confidence":     result.confidence,
                "key_factors":    result.key_factors,
                "match_preview":  result.match_preview,
                "predicted_winner": result.predicted_winner,
            })
        except Exception as e:
            predictions.append({
                "match_id":   match["id"],
                "home_team":  home_name,
                "away_team":  away_name,
                "competition": match["competition"]["name"],
                "match_date":  match["utcDate"][:10],
                "error": str(e),
            })

    return {**state, "predictions": predictions}


# --- Build graph ---

def build_pipeline():
    g = StateGraph(State)
    g.add_node("fetch_fixtures",  fetch_fixtures)
    g.add_node("enrich_matches",  enrich_matches)
    g.add_node("predict_matches", predict_matches)
    g.add_edge(START,             "fetch_fixtures")
    g.add_edge("fetch_fixtures",  "enrich_matches")
    g.add_edge("enrich_matches",  "predict_matches")
    g.add_edge("predict_matches", END)
    return g.compile()


pipeline = build_pipeline()


def run_pipeline(target_date: str) -> list[dict]:
    result = pipeline.invoke({
        "target_date": target_date,
        "matches":     [],
        "enriched":    [],
        "predictions": [],
    })
    return result.get("predictions", [])

from collections import Counter, defaultdict

def run_pipeline_averaged(target_date: str, runs: int = 3) -> list[dict]:
    all_runs = []
    for i in range(runs):
        result = run_pipeline(target_date)
        all_runs.append(result)

    # Group by match_id
    by_match = defaultdict(list)
    for run in all_runs:
        for pred in run:
            if "error" not in pred:
                by_match[pred["match_id"]].append(pred)

    averaged = []
    for match_id, preds in by_match.items():
        if not preds:
            continue
        n = len(preds)
        base = preds[0].copy()
        base["home_win_prob"] = sum(p["home_win_prob"] for p in preds) / n
        base["draw_prob"]     = sum(p["draw_prob"]     for p in preds) / n
        base["away_win_prob"] = sum(p["away_win_prob"] for p in preds) / n
        base["home_goals"]    = sum(p["home_goals"]    for p in preds) / n
        base["away_goals"]    = sum(p["away_goals"]    for p in preds) / n
        # Majority vote for winner and confidence
        base["predicted_winner"] = Counter(p["predicted_winner"] for p in preds).most_common(1)[0][0]
        base["confidence"]       = Counter(p["confidence"]       for p in preds).most_common(1)[0][0]
        averaged.append(base)

    return averaged


def refresh_actual_results():
    """Update DB with real scores for past unresolved predictions."""
    unresolved = get_unresolved_predictions()
    updated = 0
    for pred in unresolved:
        result = get_match_result(pred["match_id"])
        if result:
            update_actual_result(
                pred["match_id"],
                result["home_goals"],
                result["away_goals"],
                result["winner"]
            )
            updated += 1
    return updated
