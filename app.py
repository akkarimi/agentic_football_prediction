import html
import os
from datetime import date, datetime, timedelta

import streamlit as st

from config import FOOTBALL_DATA_API_KEY, GROQ_API_KEY, DAILY_RUN_TIME
from daily_job import ensure_scheduler, scheduler_status
from db import (
    init_db, get_predictions, get_meta,
    migrate_actual_columns, get_accuracy_stats, get_past_prediction_dates,
)
from pipeline.graph import TRACKED_TEAMS


st.set_page_config(
    page_title="Football Predictions",
    page_icon="⚽",
    layout="wide",
)

st.markdown("""
    <style>
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] { padding-top: 2rem; max-width: 1100px; }
    .fp-hero { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 1rem 2rem;
               background: linear-gradient(135deg, #053D22 0%, #0B8A3E 100%); color: #FFFFFF;
               border-radius: 1rem; padding: 1.4rem 1.8rem; margin-bottom: 0.4rem; }
    .fp-hero-title { font-size: 1.9rem; font-weight: 800; line-height: 1.2; }
    .fp-hero-sub { font-size: 0.95rem; opacity: 0.85; margin-top: 0.2rem; }
    .fp-hero-stats { display: flex; flex-wrap: wrap; gap: 0.6rem; }
    .fp-tile { background: rgba(255, 255, 255, 0.14); border-radius: 0.7rem; padding: 0.6rem 1rem;
               min-width: 8.5rem; text-align: center; }
    .fp-tile-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.85; }
    .fp-tile-value { font-size: 1.7rem; font-weight: 800; line-height: 1.25; font-variant-numeric: tabular-nums; }
    .fp-tile-sub { font-size: 0.75rem; opacity: 0.85; }
    .fp-day { font-size: 1.5rem; font-weight: 700; margin: 0.8rem 0 0; }
    .fp-day span { font-weight: 400; opacity: 0.6; font-size: 1rem; margin-left: 0.5rem; }
    .fp-comp { font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;
               opacity: 0.65; margin: 0.8rem 0 0; }
    .fp-meta { display: flex; justify-content: space-between; font-size: 0.8rem; opacity: 0.7; }
    .fp-teams { display: flex; align-items: center; gap: 1rem; margin: 0.6rem 0 0.9rem; }
    .fp-team { flex: 1; font-size: 1.35rem; font-weight: 400; opacity: 0.75; }
    .fp-team.fp-pick { font-weight: 700; opacity: 1; }
    .fp-team.fp-away { text-align: right; }
    .fp-score { text-align: center; min-width: 7rem; }
    .fp-score-value { font-size: 1.6rem; font-weight: 700; font-variant-numeric: tabular-nums; }
    .fp-score-label { font-size: 0.72rem; opacity: 0.6; text-transform: uppercase; letter-spacing: 0.05em; }
    .fp-bar { display: flex; border-radius: 0.4rem; overflow: hidden; gap: 2px; margin-bottom: 0.6rem; }
    .fp-bar > div { min-width: max-content; padding: 0.3rem 0.6rem; text-align: center; white-space: nowrap;
                    color: #FFFFFF; font-size: 0.82rem; font-variant-numeric: tabular-nums; }
    .fp-bar > .fp-pick { font-weight: 700; }
    .fp-home-seg { background: #0B8A3E; }
    .fp-draw-seg { background: #5F6673; }
    .fp-away-seg { background: #2563EB; }
    .fp-final { text-align: center; font-size: 0.95rem; margin: 0.2rem 0 0.6rem; }
    @media (max-width: 640px) {
        .fp-team { font-size: 1.05rem; }
        .fp-score { min-width: 5rem; }
        .fp-score-value { font-size: 1.25rem; }
        .fp-bar > div { padding: 0.3rem 0.35rem; font-size: 0.75rem; }
    }
    </style>
""", unsafe_allow_html=True)

init_db()

migrate_actual_columns()

# The daily job (final scores + predictions for the next two days) runs in the background
keys_ok = bool(FOOTBALL_DATA_API_KEY and GROQ_API_KEY)
if keys_ok and not os.getenv("FP_DISABLE_SCHEDULER"):
    ensure_scheduler()

# ── Header ────────────────────────────────────────────────────────────────────

def hero_html() -> str:
    def tile(label: str, value: str, sub: str) -> str:
        return (f"<div class='fp-tile'><div class='fp-tile-label'>{label}</div>"
                f"<div class='fp-tile-value'>{value}</div><div class='fp-tile-sub'>{sub}</div></div>")

    tiles = []
    for label, days in [("Last 7 days", 7), ("All time", None)]:
        stats = get_accuracy_stats(days=days)
        if stats:
            tiles.append(tile(f"{label} accuracy", f"{stats['accuracy']:.0f}%",
                              f"{stats['correct']} of {stats['total']} outcomes correct"))
        else:
            tiles.append(tile(f"{label} accuracy", "–", "no results yet"))
    if stats:
        tiles.append(tile("Goals off by", f"{(stats['home_goal_mae'] + stats['away_goal_mae']) / 2:.1f}",
                          f"home {stats['home_goal_mae']} · away {stats['away_goal_mae']}"))

    return (
        "<div class='fp-hero'>"
        "<div><div class='fp-hero-title'>⚽ Football Predictor</div>"
        "<div class='fp-hero-sub'>AI match predictions for Europe's top clubs</div></div>"
        f"<div class='fp-hero-stats'>{''.join(tiles)}</div>"
        "</div>"
    )


CONF_ICON = {"low": "🔴", "medium": "🟡", "high": "🟢"}


def kickoff_label(utc_date: str) -> str:
    try:
        return datetime.fromisoformat(utc_date.replace("Z", "+00:00")).strftime("%H:%M UTC")
    except (AttributeError, ValueError):
        return ""


def fmt_score(home_goals, away_goals) -> str:
    home_goals, away_goals = home_goals or 0, away_goals or 0
    # Older predictions stored decimal expected goals; newer ones are whole scorelines
    if float(home_goals).is_integer() and float(away_goals).is_integer():
        return f"{int(home_goals)} – {int(away_goals)}"
    return f"{home_goals:.1f} – {away_goals:.1f}"


def match_card_html(p: dict) -> str:
    home, away = html.escape(p["home_team"]), html.escape(p["away_team"])
    pick = p["predicted_winner"]
    conf = p.get("confidence") or "low"
    # Normalise so probabilities stored on a 0-100 scale still read as percentages
    total = (p["home_win_prob"] + p["draw_prob"] + p["away_win_prob"]) or 1
    probs = [(side, name, p[key] / total) for side, name, key in
             [("home", "Home", "home_win_prob"), ("draw", "Draw", "draw_prob"), ("away", "Away", "away_win_prob")]]

    def cls(base: str, side: str) -> str:
        return f"{base} fp-pick" if side == pick else base

    segments = "".join(
        f"<div class='{cls(f'fp-{side}-seg', side)}' style='flex:{prob:.4f} 1 0'>{name} {prob*100:.0f}%</div>"
        for side, name, prob in probs
    )

    parts = [
        f"<div class='fp-meta'><span>{kickoff_label(p.get('utc_date'))}</span>"
        f"<span>{CONF_ICON.get(conf, '⚪')} {html.escape(conf.capitalize())} confidence</span></div>",
        f"<div class='fp-teams'>"
        f"<div class='{cls('fp-team', 'home')}'>{home}</div>"
        f"<div class='fp-score'><div class='fp-score-value'>{fmt_score(p.get('home_goals'), p.get('away_goals'))}</div>"
        f"<div class='fp-score-label'>{'Draw predicted' if pick == 'draw' else 'Predicted score'}</div></div>"
        f"<div class='{cls('fp-team fp-away', 'away')}'>{away}</div>"
        f"</div>",
        f"<div class='fp-bar'>{segments}</div>",
    ]

    # Actual result (if available)
    actual_hg, actual_ag = p.get("actual_home_goals"), p.get("actual_away_goals")
    if actual_hg is not None and actual_ag is not None:
        correct = pick == p.get("actual_winner")
        parts.append(
            f"<div class='fp-final'>Final score <b>{int(actual_hg)} – {int(actual_ag)}</b> · "
            f"{'✅ outcome called correctly' if correct else '❌ outcome missed'}</div>"
        )
    return "".join(parts)


def render_match(p: dict):
    with st.container(border=True):

        if "error" in p:
            st.warning(f"{p['home_team']} vs {p['away_team']} — prediction failed: {p['error']}")
            return

        st.markdown(match_card_html(p), unsafe_allow_html=True)

        # Expandable analysis
        factors = p.get("key_factors", [])
        preview = p.get("match_preview", "")
        if factors or preview:
            with st.expander("Analysis"):
                if factors:
                    st.markdown("**Key factors:**")
                    for f in factors:
                        st.markdown(f"- {f}")
                if preview:
                    st.markdown("**Preview:**")
                    st.write(preview)


def render_day(predictions: list[dict]):
    by_competition: dict[str, list] = {}
    for p in predictions:
        by_competition.setdefault(p["competition"], []).append(p)

    for competition, matches in by_competition.items():
        st.markdown(f"<div class='fp-comp'>{html.escape(competition)}</div>", unsafe_allow_html=True)
        for p in sorted(matches, key=lambda m: (m.get("utc_date") or "", m["home_team"])):
            render_match(p)


def day_heading(label: str, day: date):
    st.markdown(f"<div class='fp-day'>{label}<span>{day.strftime('%A, %B %d')}</span></div>", unsafe_allow_html=True)


# Re-runs on a timer so stats and predictions written by the daily job appear without a reload
@st.fragment(run_every=60)
def page():
    st.markdown(hero_html(), unsafe_allow_html=True)

    col_status, col_about = st.columns([5, 1.2], vertical_alignment="center")
    with col_status:
        status = scheduler_status()
        finished = get_meta("last_daily_run_finished")
        if status:
            st.caption(f"⏳ Updating now — {status}")
        elif finished:
            st.caption(f"Updates automatically every day at {DAILY_RUN_TIME} · last update {finished.replace('T', ' ')}")
        else:
            st.caption(f"Updates automatically every day at {DAILY_RUN_TIME}")
    with col_about:
        with st.popover("About", use_container_width=True):
            st.markdown(
                "AI-powered match predictions using live stats, standings, "
                "and the latest news for Europe's top clubs.\n\n"
                "**Tracked teams:** " + " · ".join(TRACKED_TEAMS)
            )
            st.caption("Powered by Groq + LangGraph · Built by **AK**")

    if not keys_ok:
        st.error("API keys missing. Check your .env file.")
    errors = get_meta("last_daily_run_errors")
    if errors:
        with st.expander(f"⚠️ {len(errors.splitlines())} problem(s) in the last update"):
            st.code(errors)

    tab_upcoming, tab_history = st.tabs(["Upcoming", "History"])

    with tab_upcoming:
        today = date.today()
        for label, day in [("Today", today), ("Tomorrow", today + timedelta(days=1))]:
            day_heading(label, day)
            predictions = get_predictions(day.isoformat())
            if predictions:
                render_day(predictions)
            else:
                st.caption("No predictions for tracked teams on this day.")

    with tab_history:
        past = get_past_prediction_dates()
        if not past:
            st.caption("No past predictions yet.")
        else:
            counts = dict(past)
            chosen = st.selectbox(
                "Match day",
                list(counts),
                format_func=lambda d: f"{date.fromisoformat(d).strftime('%A, %B %d, %Y')} · {counts[d]} match(es)",
            )
            render_day(get_predictions(chosen))


page()
