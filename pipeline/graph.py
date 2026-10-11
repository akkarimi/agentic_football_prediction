from typing import TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
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
    home_win_prob:    float = Field(description="Probability the home team wins, between 0 and 1.")
    draw_prob:        float = Field(description="Probability of a draw, between 0 and 1.")
    away_win_prob:    float = Field(description="Probability the away team wins, between 0 and 1.")
    home_goals:       int = Field(description="Home team goals in the most likely final score.")
    away_goals:       int = Field(description="Away team goals in the most likely final score.")
    confidence:       Literal["low", "medium", "high"]
    key_factors:      list[str]
    match_preview: str = Field(description="A detailed 2-3 sentence match preview covering recent form, what could decide the match, and tactical outlook. Must be specific to these two teams. Only names players that appear in the supplied news.")
    predicted_winner: Literal["home", "draw", "away"]


def _consistency_problem(p: Prediction) -> str | None:
    """Describe how a prediction contradicts itself, or None if it is coherent."""
    probs = {"home": p.home_win_prob, "draw": p.draw_prob, "away": p.away_win_prob}
    if min(probs.values()) < 0 or sum(probs.values()) <= 0:
        return "The probabilities must be non-negative and sum to 1."
    if probs[p.predicted_winner] < max(probs.values()):
        return (f"predicted_winner is '{p.predicted_winner}' but that is not the outcome with the "
                f"highest probability (home {p.home_win_prob}, draw {p.draw_prob}, away {p.away_win_prob}).")
    score_winner = "home" if p.home_goals > p.away_goals else "away" if p.away_goals > p.home_goals else "draw"
    if score_winner != p.predicted_winner:
        return (f"The score {p.home_goals}-{p.away_goals} is a '{score_winner}' result but "
                f"predicted_winner is '{p.predicted_winner}'.")
    return None



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
    for item in items[:8]:
        title = item.get("title", "")
        body  = item.get("body", item.get("snippet", ""))[:150]
        day   = (item.get("date") or "")[:10]
        if title:
            lines.append(f"• {day + ' — ' if day else ''}{title}: {body}")
    return "\n".join(lines)


# --- Nodes ---

# Display name -> lowercase fragments that identify the club in Football-Data.org team names.
# The page's team list is generated from this, so add or remove clubs here only.
TRACKED_TEAMS = {
    "Manchester United":   ["manchester united"],
    "Manchester City":     ["manchester city"],
    "Arsenal":             ["arsenal"],
    "Chelsea":             ["chelsea"],
    "Liverpool":           ["liverpool"],
    "Tottenham Hotspur":   ["tottenham"],
    "Real Madrid":         ["real madrid"],
    "Barcelona":           ["fc barcelona"],
    "Atletico Madrid":     ["atletico madrid", "atlético de madrid"],
    "Bayern Munich":       ["bayern"],
    "Borussia Dortmund":   ["dortmund"],
    "Inter Milan":         ["inter milan", "internazionale"],
    "AC Milan":            ["ac milan"],
    "Juventus":            ["juventus"],
    "Napoli":              ["napoli"],
    "Paris Saint-Germain": ["paris saint-germain", "psg"],
}

def _is_tracked(match: dict) -> bool:
    home = match.get("homeTeam", {}).get("name", "").lower()
    away = match.get("awayTeam", {}).get("name", "").lower()
    return any(alias in home or alias in away for aliases in TRACKED_TEAMS.values() for alias in aliases)


def fetch_tracked_matches(target_date: str) -> list[dict]:
    """Tracked-team fixtures for a date, in kickoff order."""
    matches = [m for m in get_todays_matches(target_date) if _is_tracked(m)]
    return sorted(matches, key=lambda m: m.get("utcDate", ""))


def fetch_fixtures(state: State) -> State:
    return {**state, "matches": fetch_tracked_matches(state["target_date"])}


def _enrich_match(match: dict) -> dict:
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

        # Two searches per team: availability, then the wider story around the club (mood, manager, off-field)
        home_news, away_news = (
            search_news(f"{name} injury team news")
            + search_news(f"{name} football news")
            for name in (home_name, away_name)
        )

        return {
            **match,
            "home_form": home_form,
            "away_form": away_form,
            "h2h":       h2h,
            "standings": standings,
            "home_news": home_news,
            "away_news": away_news,
        }
    except Exception as e:
        return {**match, "enrich_error": str(e)}


def enrich_matches(state: State) -> State:
    return {**state, "enriched": [_enrich_match(m) for m in state["matches"]]}


def _structured_llm():
    llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.1)
    # The model occasionally returns malformed JSON, so retry before giving up on a match
    return llm.with_structured_output(Prediction, method="json_schema").with_retry(stop_after_attempt=3)


def _predict_match(structured_llm, match: dict) -> dict:
    home_name = match["homeTeam"]["name"]
    away_name = match["awayTeam"]["name"]
    base = {
        "match_id":    match["id"],
        "home_team":   home_name,
        "away_team":   away_name,
        "competition": match["competition"]["name"],
        "match_date":  match["utcDate"][:10],
    }

    if "enrich_error" in match:
        return {**base, "error": match["enrich_error"]}

    prompt = build_match_prompt(
        home_team   = home_name,
        away_team   = away_name,
        competition = match["competition"]["name"],
        match_date  = match["utcDate"][:10],
        home_form   = _fmt_form(match.get("home_form", [])),
        away_form   = _fmt_form(match.get("away_form", [])),
        h2h         = _fmt_h2h(match.get("h2h", {})),
        standings   = _fmt_standings(match.get("standings", {}), home_name, away_name),
        news        = (f"{home_name}:\n{_fmt_news(match.get('home_news', []))}\n\n"
                       f"{away_name}:\n{_fmt_news(match.get('away_news', []))}"),
    )

    try:
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
        # Winner, probabilities and score come back as separate fields and can disagree,
        # so send a contradictory answer back for correction rather than showing it
        for _ in range(3):
            result = structured_llm.invoke(messages)
            problem = _consistency_problem(result)
            if not problem:
                break
            messages += [
                AIMessage(content=result.model_dump_json()),
                HumanMessage(content=f"That prediction contradicts itself. {problem} Give a corrected, consistent prediction."),
            ]
        else:
            return {**base, "error": f"Model kept giving a contradictory prediction. {problem}"}

        total = result.home_win_prob + result.draw_prob + result.away_win_prob
        return {
            **base,
            "utc_date":       match["utcDate"],
            "home_win_prob":  result.home_win_prob / total,
            "draw_prob":      result.draw_prob / total,
            "away_win_prob":  result.away_win_prob / total,
            "home_goals":     result.home_goals,
            "away_goals":     result.away_goals,
            "confidence":     result.confidence,
            "key_factors":    result.key_factors,
            "match_preview":  result.match_preview,
            "predicted_winner": result.predicted_winner,
        }
    except Exception as e:
        return {**base, "error": str(e)}


def predict_matches(state: State) -> State:
    structured_llm = _structured_llm()
    return {**state, "predictions": [_predict_match(structured_llm, m) for m in state["enriched"]]}


def predict_match(match: dict) -> dict:
    """Enrich and predict a single fixture, so callers can show results one match at a time."""
    return _predict_match(_structured_llm(), _enrich_match(match))


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
    failed = {}
    for run in all_runs:
        for pred in run:
            if "error" not in pred:
                by_match[pred["match_id"]].append(pred)
            else:
                failed[pred["match_id"]] = pred

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

    # Matches that failed in every run are returned with their error so the caller can report them
    averaged.extend(p for match_id, p in failed.items() if match_id not in by_match)

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
